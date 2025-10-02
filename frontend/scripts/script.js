const myButton = document.querySelector("button")

const myTextArea = document.querySelector("input")

const myDiv = document.getElementById("results")

const myURL = "http://0.0.0.0:8000"

function search_by_id() {
    arxiv_id = myTextArea.value

    myDiv.innerHTML = `<p class="text-gray-500">Searching...</p>`;

    // Call `fetch()`, passing in the URL.
    fetch("http://0.0.0.0:8000/search_by_id/"  + arxiv_id)
    // fetch() returns a promise. When we have received a response from the server,
    // the promise's `then()` handler is called with the response.
    .then((response) => {
        // Our handler throws an error if the request did not succeed.
        if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
        }
        // Otherwise (if the response succeeded), our handler fetches the response
        // as text by calling response.text(), and immediately returns the promise
        // returned by `response.text()`.
        return response.json();
    })
    // When response.text() has succeeded, the `then()` handler is called with
    // the text, and we copy it into the `poemDisplay` box.
    .then((json) => {
        myDiv.innerHTML = json.map(result => `
                    <div>
                    <a href="${result.entity.url}" target="_blank">
                       <h2 id="results_title"> ${result.entity.title} </h2>
                    </a>
                    <b> ${result.entity.authors} </b>  | <i>${result.entity.month} ${result.entity.year}</i>
                    <p>${result.entity.abstract}...</p>
                    <p>
                    <em>${result.entity.categories}</em>
                    </p>

                    </div>`).join("");

    // Render all math expressions after content is loaded
    renderMathInElement(myDiv, {
        delimiters: [
                            {left: '$$', right: '$$', display: true},
                            {left: '$', right: '$', display: false}],
                        throwOnError: false
                    });
    })
    // Catch any errors that might happen, and display a message
    // in the `poemDisplay` box.
    .catch((error) => {
        myDiv.textContent = `Could not fetch verse: ${error}`;
    });
    }

myButton.addEventListener("click", search_by_id)