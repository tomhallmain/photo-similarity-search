# Local Image Similarity Search App

Forked version of the original project by [@harperreed](https://github.com/harperreed).

The following are changes from the original:
- Expands image file types to a configurable list
- Adds a masonry layout to the UI (Macy)
- Makes number of result images in UI configurable
- Enables the application to use multiple directories of images in one database
- Enables CLIP embedding handling for other platforms using CLIP python module
- Makes CLIP model configurable
- Adds special handling for resizing images in UI


# Original Project README

# 📸 Embed-Photos 🖼️

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Welcome to Embed-Photos, a powerful photo similarity search engine built by [@harperreed](https://github.com/harperreed)! 🎉 This project leverages the CLIP (Contrastive Language-Image Pre-training) model to find visually similar images based on textual descriptions. 🔍🖼️

## 🌟 Features

- 🚀 Fast and efficient image search using the CLIP model
- 💻 Works on Apple Silicon (MLX) ~~only~~ **as well as other platforms**
- 💾 Persistent storage of image embeddings using SQLite and Chroma
- 🌐 Web interface for easy interaction and exploration
- 🔒 Secure image serving and handling
- 📊 Logging and monitoring for performance analysis
- 🔧 Configurable settings using ~~environment variables~~ **JSON config files**

## Screenshot

![image](https://github.com/harperreed/photo-similarity-search/assets/18504/7df51659-84b0-4efb-9647-58a544743ea5)


## 📂 Repository Structure

```
embed-photos/
├── README.md
├── generate_embeddings.py
├── requirements.txt
├── start_web.py
└── templates
    ├── README.md
    ├── base.html
    ├── display_image.html
    ├── index.html
    ├── output.txt
    └── query_results.html
```

- `generate_embeddings.py`: Script to generate image embeddings using the CLIP model
- `requirements.txt`: Lists the required Python dependencies
- `start_web.py`: Flask web application for the photo similarity search
- `templates/`: Contains HTML templates for the web interface

## 🚀 Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/harperreed/photo-similarity-search.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the application by setting the necessary environment variables in a `.env` file.

4. Generate image embeddings:
   ```
   python generate_embeddings.py
   ```

5. Start the web application:
   ```
   python start_web.py
   ```

6. Open your web browser and navigate to `http://localhost:5000` to explore the photo similarity search!

## Todo

- Use siglip instead of clip
- add a more robust config
- make mlx optional

## 🙏 Acknowledgments

The Embed-Photos project builds upon the work of the Apple (mlx!), the CLIP model and leverages various open-source libraries. We extend our gratitude to the authors and contributors of these projects.

Happy searching! 🔍✨
