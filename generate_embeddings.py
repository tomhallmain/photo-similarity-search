import os
import msgpack
import time
from dotenv import load_dotenv
import sqlite3
import signal
import hashlib
from concurrent.futures import ThreadPoolExecutor
import chromadb

from config import config
from log_config import get_logger
from model import image_embeddings

# Configure logging
logger, log_level = get_logger("app")
config.log(logger)

# Graceful shutdown handler
def graceful_shutdown(signum, frame):
    logger.info("Caught signal, shutting down gracefully...")
    if 'conn_pool' in globals():
        connection.close()
        logger.info("Database connection pool closed.")
    exit(0)

# Register the signal handlers for graceful shutdown
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

# Load environment variables
load_dotenv()

# Check if data dir exists, if it doesn't - then create it
if not os.path.exists(config.DATA_DIR):
    logger.info("Creating data directory ...")
    os.makedirs(config.DATA_DIR)
    
# Create a connection pool for the SQLite database
connection = sqlite3.connect(config.SQLITE_DB_FILEPATH)

def create_table():
    """
    Creates the 'images' table in the SQLite database if it doesn't exist.
    """
    with connection:
        connection.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_date TEXT NOT NULL,
                file_md5 TEXT NOT NULL,
                embeddings BLOB
            )
        ''')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_filename ON images (filename)')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON images (file_path)')
    logger.info("Table 'images' ensured to exist.")



def file_generator(directories):
    """
    Generates file paths for all files in the specified directory and its subdirectories.

    :param directory: The directory path to search for files.
    :return: A generator yielding file paths.
    """
    for directory in directories:
        logger.debug(f"Generating file paths for directory: {directory}")
        for root, _, files in os.walk(directory):
            for file in files:
                yield os.path.join(root, file)

def hydrate_cache(directories, cache_file_path):
    """
    Loads or generates a cache of file paths for the specified directory.

    :param directory: The directory path to search for files.
    :param cache_file_path: The path to the cache file.
    :return: A list of cached file paths.
    """
    logger.info(f"Hydrating cache for {directories} using {cache_file_path}...")
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'rb') as f:
                cached_files = msgpack.load(f)
            logger.info(f"Loaded cached files from {cache_file_path}")
            if len(cached_files) == 0:
                logger.warning(f"Cache file {cache_file_path} is empty. Regenerating cache...")
                cached_files = list(file_generator(directories))
                with open(cache_file_path, 'wb') as f:
                    msgpack.dump(cached_files, f)
                logger.info(f"Regenerated cache with {len(cached_files)} files and dumped to {cache_file_path}")
        except (msgpack.UnpackException, IOError) as e:
            logger.error(f"Error loading cache file {cache_file_path}: {e}. Regenerating cache...")
            cached_files = list(file_generator(directories))
            with open(cache_file_path, 'wb') as f:
                msgpack.dump(cached_files, f)
            logger.info(f"Regenerated cache with {len(cached_files)} files and dumped to {cache_file_path}")
    else:
        logger.info(f"Cache file not found at {cache_file_path}. Creating cache dirlist for {directories}...")
        cached_files = list(file_generator(directories))
        try:
            with open(cache_file_path, 'wb') as f:
                msgpack.dump(cached_files, f)
            logger.info(f"Created cache with {len(cached_files)} files and dumped to {cache_file_path}")
        except IOError as e:
            logger.error(f"Error creating cache file {cache_file_path}: {e}. Proceeding without cache.")
    return cached_files


def update_db(image):
    """
    Updates the database with the image embeddings.

    :param image: A dictionary containing image information.
    """
    try:
        embeddings_blob = sqlite3.Binary(msgpack.dumps(image.get('embeddings', [])))
        with sqlite3.connect(config.SQLITE_DB_FILEPATH) as conn:
            conn.execute("UPDATE images SET embeddings = ? WHERE filename = ?",
                         (embeddings_blob, image['filename']))
        logger.debug(f"Database updated successfully for image: {image['filename']}")
    except sqlite3.Error as e:
        logger.error(f"Database update failed for image: {image['filename']}. Error: {e}")

def process_image(file_path):
    """
    Processes an image file by extracting metadata and inserting it into the database.

    :param file_path: The path to the image file.
    """
    file = os.path.basename(file_path)
    file_date = time.ctime(os.path.getmtime(file_path))
    with open(file_path, 'rb') as f:
        file_content = f.read()
    file_md5 = hashlib.md5(file_content).hexdigest()
    conn = None
    try:
        conn = sqlite3.connect(config.SQLITE_DB_FILEPATH)
        with conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT EXISTS(SELECT 1 FROM images WHERE filename=? AND file_path=? LIMIT 1)
            ''', (file, file_path))
            result = cursor.fetchone()
            file_exists = result[0] if result else False
            if not file_exists:
                cursor.execute('''
                    INSERT INTO images (filename, file_path, file_date, file_md5)
                    VALUES (?, ?, ?, ?)
                ''', (file, file_path, file_date, file_md5))
                logger.debug(f'Inserted {file} with metadata into the database.')
            else:
                logger.debug(f'File {file} already exists in the database. Skipping insertion.')
    except sqlite3.Error as e:
        logger.error(f'Error processing image {file}: {e}')
    finally:
        if conn:
            conn.close()

def process_embeddings(photo):
    """
    Processes image embeddings by uploading them to the embedding server and updating the database.

    :param photo: A dictionary containing photo information.
    """
    logger.debug(f"Processing photo: {photo['filename']}")
    if photo['embeddings']:
        logger.debug(f"Photo {photo['filename']} already has embeddings. Skipping.")
        return

    try:
        start_time = time.time()
        imemb = image_embeddings(photo['file_path'])
        photo['embeddings'] = imemb
        update_db(photo)
        end_time = time.time()
        logger.debug(f"Processed embeddings for {photo['filename']} in {end_time - start_time:.5f} seconds")
    except Exception as e:
        logger.error(f"Error generating embeddings for {photo['filename']}: {e}")


def main():
    """
    Main function to process images and embeddings.
    """
    cache_start_time = time.time()
    cached_files = hydrate_cache(config.SOURCE_IMAGE_DIRECTORIES, config.FILELIST_CACHE_FILEPATH)
    cache_end_time = time.time()
    logger.info(f"Cache operation took {cache_end_time - cache_start_time:.2f} seconds")
    logger.info(f"Directory has {len(cached_files)} files: {config.SOURCE_IMAGE_DIRECTORIES}")

    create_table()

    with ThreadPoolExecutor() as executor:
        futures = []
        for file_path in cached_files:
            for extension in config.FILE_TYPES:
                if file_path.lower().endswith("." + extension):
                    future = executor.submit(process_image, file_path)
                    futures.append(future)
                    break
        for future in futures:
            future.result()

    with connection:
        cursor = connection.cursor()
        cursor.execute("SELECT filename, file_path, file_date, file_md5, embeddings FROM images")
        photos = [{'filename': row[0], 'file_path': row[1], 'file_date': row[2], 'file_md5': row[3], 'embeddings': msgpack.loads(row[4]) if row[4] else []} for row in cursor.fetchall()]
        # for photo in photos:
        #     photo['embeddings'] = msgpack.loads(photo['embeddings']) if photo['embeddings'] else []

    num_photos = len(photos)

    logger.info(f"Loaded {num_photos} photos from database")
    #cant't use ThreadPoolExecutor here because of the MLX memory thing
    start_time = time.time()
    photo_ite = 0
    for photo in photos:
        process_embeddings(photo)
        photo_ite += 1
        if log_level != 'DEBUG':
            if photo_ite % 100 == 0:
                logger.info(f"Processed {photo_ite}/{num_photos} photos")
    end_time = time.time()
    logger.info(f"Generated embeddings for {len(photos)} photos in {end_time - start_time:.2f} seconds")
    connection.close()
    logger.info("Database connection pool closed.")


    logger.info(f"Initializing Chrome DB:  {config.CHROMA_COLLECTION_NAME}")
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)

    logger.info(f"Generated embeddings for {len(photos)} photos")
    start_time = time.time()

    photo_ite = 0
    for photo in photos:
        # Skip processing if the photo does not have embeddings
        if not photo['embeddings']:
            logger.debug(f"[{photo_ite}/{num_photos}] Photo {photo['filename']} has no embeddings. Skipping addition to Chroma.")
            continue

        try:
            # Add the photo's embeddings to the Chroma collection
            item = collection.get(ids=[photo['filename']])
            if item['ids'] !=[]:
                continue
            collection.add(
                embeddings=[photo["embeddings"]],
                documents=[photo['filename']],
                ids=[photo['filename']]
            )
            logger.debug(f"[{photo_ite}/{num_photos}] Added embedding to Chroma for {photo['filename']}")
            photo_ite += 1
            if log_level != 'DEBUG':
                if photo_ite % 100 == 0:
                    logger.info(f"Processed {photo_ite}/{num_photos} photos")
        except Exception as e:
            # Log an error if the addition to Chroma fails
            logger.error(f"[{photo_ite}/{num_photos}] Failed to add embedding to Chroma for {photo['filename']}: {e}")
    end_time = time.time()
    logger.info(f"Inserted embeddings {len(photos)} photos into Chroma in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
