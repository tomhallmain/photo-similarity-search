import os
import random
import signal
import sqlite3
from dotenv import load_dotenv
from flask import jsonify, g, send_file
from flask import Flask, render_template, request, redirect, url_for
from io import BytesIO
import chromadb
from PIL import Image, ImageOps

from config import config
from log_config import get_logger
from model import text_embeddings

# Configure logging
logger, log_level = get_logger("web")
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

# Create a connection pool for the SQLite database
connection = sqlite3.connect(config.SQLITE_DB_FILEPATH)

app = Flask(__name__)

#Instantiate MLX Clip model
#clip = mlx_clip.mlx_clip("mlx_model", hf_repo=config.CLIP_MODEL)

logger.info(f"Initializing Chrome DB:  {config.CHROMA_COLLECTION_NAME}")
client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)
items = collection.get()["ids"]


def get_file_path_from_db(filename):
    """
    Fetch the full file path from the database for a given filename.
    
    :param filename: The name of the file to look up.
    :return: The full file path of the file, or None if not found.
    """
    # establish connection to database:
    conn = sqlite3.connect(config.SQLITE_DB_FILEPATH)
    cursor = conn.cursor()

    # Define a SQL query to retrieve the file_path from the 'images' table
    query = f"SELECT file_path FROM images WHERE filename = {repr(filename)} LIMIT 1"
    cursor.execute(query)

    # Fetch result
    full_path = None
    result = cursor.fetchone()
    if result:
        full_path = result[0]
    cursor.close()
    conn.close()
    return full_path

# WEBS


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


@app.route("/")
def index():
    images = collection.get()["ids"]
    n_images_to_get = min(len(images), config.NUM_IMAGE_RESULTS)
    random_items = random.sample(images, n_images_to_get)
    # Display a form or some introduction text
    return render_template("index.html", images=random_items)


@app.route("/image/<filename>")
def serve_specific_image(filename):
    # Construct the filepath and check if it exists
    print(filename)

    filepath = get_file_path_from_db(filename)
    if filepath is None or not os.path.exists(filepath):
        return f"Image not found: {filename}", 404

    image = collection.get(ids=[filename], include=["embeddings"])
    results = collection.query(
        query_embeddings=image["embeddings"], n_results=(config.NUM_IMAGE_RESULTS + 1)
    )

    for i in range(len(results["ids"][0])):
        distance = int(results["distances"][0][i])
        f = results["ids"][0][i]
        print(f"{distance} - {f}")

    images = []
    for ids in results["ids"]:
        for id in ids:
            # Adjust the path as needed
            image_url = url_for("serve_image", filename=id)
            images.append({"url": image_url, "id": id})

    # Use the proxy function to serve the image if it exists
    image_url = url_for("serve_image", filename=filename, resize=False)

    # Render the template with the specific image
    return render_template("display_image.html", image=image_url, images=images[1:])


@app.route("/random")
def random_image():
    images = collection.get()["ids"]
    image = random.choice(images) if images else None

    if image:
        return redirect(url_for("serve_specific_image", filename=image))
    else:
        return "No images found", 404


@app.route("/text-query", methods=["GET"])
def text_query():

    # Assuming there's an input for embeddings; this part is tricky and needs customization
    # You might need to adjust how embeddings are received or generated based on user input
    text = request.args.get("text")  # Adjusted to use GET parameters

    # Use the Clip model to generate embeddings from the text
    embeddings = text_embeddings(text)
    results = collection.query(query_embeddings=embeddings, n_results=(config.NUM_IMAGE_RESULTS))
    images = []
    for ids in results["ids"]:
        for id in ids:
            # Adjust the path as needed
            image_url = url_for("serve_image", filename=id)
            images.append({"url": image_url, "id": id})

    return render_template(
        "query_results.html", images=images, text=text, title="Text Query Results"
    )


def resize_image(img, width, height):
    # Resize the image to half the original size
    img.thumbnail((width, height))
    img = ImageOps.exif_transpose(img)
    # Save the resized image to a BytesIO object
    img_io = BytesIO()
    try:
        img.save(img_io, 'JPEG', quality=85)
    except OSError as e:
        try:
            img.save(img_io, 'PNG')
        except OSError as e0:
            raise OSError(f"Failed to resize image: {e0}")
    img_io.seek(0)
    return img_io

def scale_image(img, width, height):
    min_width = 400
    if width < min_width:
        ratio = min_width / width
        height = height * ratio
        return resize_image(img, min_width, height)
    else:
        return resize_image(img, width, height)


@app.route("/img/<path:filename>")
def serve_image(filename, resize=True):
    """
    Serve a resized image directly from the filesystem outside of the static directory.
    """

    # Construct the full file path. Be careful with security implications.
    # Ensure that you validate `filename` to prevent directory traversal attacks.
    filepath = get_file_path_from_db(filename)
    if filepath is None or not os.path.exists(filepath):
        # You can return a default image or a 404 error if the file does not exist.
        return f"Image not found: {filename}", 404

    if resize:
        with Image.open(filepath) as img:
            img_io = scale_image(img, img.width // 2, img.height // 2)
            return send_file(img_io, mimetype='image/jpeg')

    return send_file(filepath)



if __name__ == "__main__":
    if config.ENABLE_EXTERNAL_CONNECTIONS:
        app.run(debug=True, host="0.0.0.0")
        print("Flask app is exposed to external connections. Set config option ENABLE_EXTERNAL_CONNECTION to false to change this.")
    else:
        print("Flask app will not be exposed to external connections. Set config option ENABLE_EXTERNAL_CONNECTION to true to change this.")
        app.run(debug=True)
