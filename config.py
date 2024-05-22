import json
import os
import socket
import uuid


class Config:
    CONFIG_FILE_LOC = os.path.join(os.path.abspath(os.path.dirname(__file__)), "configs", "config.json")

    def __init__(self):
        self._dict = {}
        if not os.path.exists(Config.CONFIG_FILE_LOC):
            print(Config.CONFIG_FILE_LOC)
            raise Exception("Please ensure config file is named configs/config.json.")
        self._dict = json.load(open(Config.CONFIG_FILE_LOC, "r"))

        # Generate unique ID for the machine
        self.host_name = socket.gethostname()
        self.unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, self.host_name + str(uuid.getnode()))
        self.unique_id = "a9d9bc7f-1ff4-5b40-b9d2-db08cad7a42e" # TODO fix

        self.DATA_DIR = "./data"
        self.SQLITE_DB_FILENAME = "images.db"
        self.FILELIST_CACHE_FILENAME = "filelist_cache.msgpack"
        self.SOURCE_IMAGE_DIRECTORIES = []
        self.CHROMA_COLLECTION_NAME = "images"
        self.NUM_IMAGE_RESULTS = 52
        self.CLIP_MODEL = "ViT-B/32"
        self.FILE_TYPES = [".jpg", ".jpeg", ".png", ".webp"]

        self.set_values(str,
                        "CLIP_MODEL",
                        "FILE_TYPES",
                        "DATA_DIR",
                        "DB_FILENAME",
                        "CACHE_FILENAME",
                        "CHROMA_COLLECTION_NAME")
        self.set_values(int,
                        "NUM_IMAGE_RESULTS")
        self.set_values(list,
                        "FILE_TYPES",
                        "SOURCE_IMAGE_DIRECTORIES")

        self.CHROMA_DB_PATH = os.path.join(self.DATA_DIR, f"{self.unique_id}_chroma")

        # Append the unique ID to the db file path and cache file path
        self.SQLITE_DB_FILEPATH = os.path.join(self.DATA_DIR, f"{str(self.unique_id)}_{self.SQLITE_DB_FILENAME}")
        self.FILELIST_CACHE_FILEPATH = os.path.join(self.DATA_DIR, f"{self.unique_id}_{self.FILELIST_CACHE_FILENAME}")


    def log(self, logger):
        logger.info(f"Running on machine ID: {self.unique_id}")
        logger.debug("Configuration loaded.")
        # Log the configuration for debugging
        logger.debug(f"Configuration - self.DATA_DIR: {self.DATA_DIR}")
        logger.debug(f"Configuration - DB_FILENAME: {self.SQLITE_DB_FILENAME}")
        logger.debug(f"Configuration - CACHE_FILENAME: {self.FILELIST_CACHE_FILENAME}")
        logger.debug(f"Configuration - self.SOURCE_IMAGE_DIRECTORIES: {self.SOURCE_IMAGE_DIRECTORIES}")
        logger.debug(f"Configuration - CHROME_PATH: {self.CHROMA_DB_PATH}")
        logger.debug(f"Configuration - CHROME_COLLECTION: {self.CHROMA_COLLECTION_NAME}")
        logger.debug(f"Configuration - self.NUM_IMAGE_RESULTS: {self.NUM_IMAGE_RESULTS}")
        logger.debug(f"Configuration - self.CLIP_MODEL: {self.CLIP_MODEL}")
        logger.debug("Configuration loaded.")

    def set_values(self, type, *names):
        for name in names:
            if type:
                try:
                    setattr(self, name, type(self._dict[name]))
                except Exception as e:
                    print(e)
                    print(f"Failed to set {name} from config.json file. Ensure the value is set and of the correct type.")
            else:
                try:
                    setattr(self, name, self._dict[name])
                except Exception as e:
                    print(e)



config = Config()
