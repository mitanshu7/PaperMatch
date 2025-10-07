const myButton = document.getElementById("search-button")

const myTextArea = document.getElementById("search-input")

const myDiv = document.getElementById("results")

const myURL = "http://0.0.0.0:8000/search_by_id/"

function search_by_id() {
    const arxiv_id = myTextArea.value.trim()

    myDiv.innerHTML = `<p></p>
                      <div id="loader"></div>`;

    // Call `fetch()`, passing in the URL.
    fetch(myURL  + arxiv_id)
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

// From https://www.w3schools.com/howto/howto_js_trigger_button_enter.asp
// Execute a function when the user presses a key on the keyboard
myTextArea.addEventListener("keydown", function(event) {
  // If the user presses the "Enter" key on the keyboard
  if (event.key === "Enter" && !event.shiftKey) {
    // Cancel the default action, if needed
    event.preventDefault();
    // Trigger the button element with a click
    myButton.click();
  }
}); 

// From https://stackoverflow.com/questions/2803880/is-there-a-way-to-get-a-textarea-to-stretch-to-fit-its-content-without-using-php
// Resize textarea
function resize() {
  myTextArea.style.height = "";
  myTextArea.style.height = myTextArea.scrollHeight + "px"
}
myTextArea.addEventListener("input", resize)