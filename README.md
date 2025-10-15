# [PaperMatch](https://papermatch.me/): arXiv Search with Embeddings and Milvus
## Backend at [embed_arxiv_simpler](https://github.com/mitanshu7/embed_arxiv_simpler)

This project allows users to search for arXiv papers either by ID or abstract. The search functionality is powered by a machine learning embedding model and Milvus, a vector database. Gradio is used to create a user-friendly web interface for interaction. 

See implemented demo at [papermatch.me](https://papermatch.me/)

![Demo](demo.gif)

See full explanation at the corresponding blog post: [mitanshu.tech/posts/papermatch](https://mitanshu.tech/posts/papermatch/)

## Features

- **Search by Abstract:** Convert the abstract into a vector and find similar papers based on cosine similarity.
- **Search by ID:** Retrieve information directly by arXiv ID.
- **Top K Results:** Display the top K results from Milvus based on similarity.
- **Embedding Model:** The embedding model used is [**mixedbread-ai/mxbai-embed-large-v1**](https://www.mixedbread.ai/docs/embeddings/mxbai-embed-large-v1) which happens to have [these nice properties](https://www.mixedbread.ai/blog/binary-mrl).

## Requirements

- Python 3.10+
- [Gradio](https://www.gradio.app/) for Frontend.
- [Milvus](https://milvus.io/) for Vector similarity search.
- [node.js](https://nodejs.org/en/download/package-manager) for SSR.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mitanshu7/PaperMatch.git
   cd PaperMatch
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

## Usage


1. **Setup app.py :**
- If using API to create embeddings, keep `LOCAL=False`:
   - Get your key from [Mixedbread](https://www.mixedbread.ai/api-reference/authentication)
   and paste it in `.env` file. See `.env.sample` for config.
- Keep `FLOAT=True` if you want to use float32 embeddings, else it will use binary embeddings.

2. **Run the Gradio app:**

   ```bash
   python app.py
   ```

3. **Interact with the web interface:**

   - Open your web browser and go to `http://localhost:7860` to access the Gradio interface.
   - Use the search bar to input arXiv ID or abstract and view the search results.


## Example

Here is a basic example of how to use the search feature:

1. **Search by Abstract:**
   - Enter the abstract of the paper in the provided text box.
   - The system will convert it to a vector, query Milvus, and return the most relevant papers.

2. **Search by ID:**
   - Input an arXiv ID directly.
   - Retrieve and display the corresponding paper details.
  
## Run at startup (systemd):
1. Create folder using `mkdir -p ~/.config/systemd/user/` if it doesn't already exist.
2. Create a service file using:
`nano ~/.config/systemd/user/papermatch.service`
with the following contents (assuming using *uv package manager*):
```bash
[Unit]
Description=PaperMatch App
After=network.target

[Service]
WorkingDirectory=/home/mitanshu/PaperMatch/
ExecStart=/home/mitanshu/.local/bin/uv run app.py
Restart=always

[Install]
WantedBy=default.target
```
replace `mitanshu` with your `username`.

1. Issue `systemctl --user daemon-reload` to reload systemd.
2. Issue `systemctl --user start papermatch.service` to start the app.
3. Issue `systemctl --user enable  papermatch.service` to enable app at start up.



## Contributing

Feel free to contribute to the project by submitting issues, pull requests, or suggestions. 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact [mitanshu.sukhwani@gmail.com](mailto:mitanshu.sukhwani@gmail.com).

## Acknowledgements

Mnay thanks to Devan for suggesting amazing fonts, Madhu for giving quality of life improvements, and Kshitij for always being the test subject. 