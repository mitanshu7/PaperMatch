# PaperMatch: arXiv Search with Embeddings and Milvus

This project allows users to search for arXiv papers either by ID or abstract. The search functionality is powered by a machine learning embedding model and Milvus, a vector database. Gradio is used to create a user-friendly web interface for interaction.

## Features

- **Search by Abstract:** Convert the abstract into a vector using the `mixedbread-ai/mxbai-embed-large-v1` model and find similar papers based on cosine similarity.
- **Search by ID:** Retrieve information directly by arXiv ID.
- **Top K Results:** Display the top K results from Milvus based on similarity.

## Requirements

- Python 3.7+
- Gradio
- Milvus
- `mixedbread-ai/mxbai-embed-large-v1` (or any compatible embedding model)

## Installation

1. **Clone the repository:**

   ```bash
   git clone [<repository-url>](https://github.com/mitanshu7/search_arxiv.git)
   cd search_arxiv
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Milvus:**
   - Follow the [Milvus installation guide](https://milvus.io/docs) to get Milvus up and running.
   - Configure Milvus with your preferred settings.
   - Or use `standalone_embed.sh` in this repo made compatible with Fedora.

## Usage

1. **Prepare Milvus:**

   ```bash
   # Command to prepare Milvus 
   python prepare_milvus.py
   ```

2. **Setup API key :**
   Get your key from [Mixedbread](https://www.mixedbread.ai/)
   and paste it in `.env` file. See `.env.sample` for config.

3. **Run the Gradio app:**

   ```bash
   python app.py
   ```

4. **Interact with the web interface:**

   - Open your web browser and go to `http://localhost:7860` to access the Gradio interface.
   - Use the search bar to input arXiv ID or abstract and view the search results.

## Configuration

- **Embedding Model:** The embedding model used is `mixedbread-ai/mxbai-embed-large-v1`.

## Example

Here is a basic example of how to use the search feature:

1. **Search by Abstract:**
   - Enter the abstract of the paper in the provided text box.
   - The system will convert it to a vector, query Milvus, and return the most relevant papers.

2. **Search by ID:**
   - Input an arXiv ID directly.
   - Retrieve and display the corresponding paper details.
  
## Run at startup (systemd):
1. create a file `~/.config/systemd/user/search_arxiv.service` using:
`nano ~/.config/systemd/user/search_arxiv.service`
with the following contents (assuming user=milvus, and using anaconda package manager with env name search_arxiv):
```bash
[Unit]
Description=Search ArXiv  Web App
After=network.target

[Service]
WorkingDirectory=/home/milvus/search_arxiv/
ExecStart=/bin/bash -c "source /home/milvus/miniforge3/bin/activate search_arxiv && python app.py"
Restart=always

[Install]
WantedBy=default.target
```
2. Issue `systemctl --user daemon-reload` to reload systemd.
3. issue `systemctl --user start search_arxiv.service` to start the app.
4. Issue `systemctl --user enable  search_arxiv.service` to enable app at start up.

## Contributing

Feel free to contribute to the project by submitting issues, pull requests, or suggestions. 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact [mitanshu.sukhwani@gmail.com](mailto:mitanshu.sukhwani@gmail.com).
