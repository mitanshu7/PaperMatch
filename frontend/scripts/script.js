// Search button
const myButton = document.getElementById("search-button")

// Search input
const myTextArea = document.getElementById("search-input")

const myDropdown = document.getElementById("year_filter")

// Search results
const myDiv = document.getElementById("results")

// Search url
const search_url = "http://0.0.0.0:8000/search"

// get current year
// https://stackoverflow.com/questions/4562587/shortest-way-to-print-current-year-in-a-website
const currentYear = new Date().getFullYear();

// Update the dropdown options
const yearDropdown = document.getElementById("year_filter");
yearDropdown.innerHTML = `
    <option value="" selected>All years</option>
    <option value="year == ${currentYear}">This year</option>
    <option value="year >= ${currentYear - 5}">Last 5 years</option>
    <option value="year >= ${currentYear - 10}">Last 10 years</option>
`;

//  Function to perform the search and modify div to render html
function search(text, filter) {

    // Show a spinning circle till the results load
    myDiv.innerHTML = `<p></p>
                      <div id="loader"></div>`;

    // From https://www.freecodecamp.org/news/javascript-post-request-how-to-send-an-http-post-request-in-js/
    fetch(search_url, {
      method: "POST",
      body: JSON.stringify({
        text: text,
        filter: filter,
      }),
      headers: {
        "Content-type": "application/json; charset=UTF-8"
      }
    })
    .then((response) => {
        if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
        }

        return response.json();
    })
    .then((json) => {
        myDiv.innerHTML = json.map(result => `
                    <div>
                    <a href="${result.entity.url}" target="_blank">
                       <h2 id="results_title"> ${result.entity.title} </h2>
                    </a>
                    <p class="hfill">
                    <b> ${result.entity.authors} </b> <i>${result.entity.month} ${result.entity.year}</i>
                    </p>
                    <p>
                    ${result.entity.abstract}...
                    </p>
                    <p class="hfill">
                    <em>${result.entity.categories}</em> <a href="?arxiv_id=${result.entity.id}"> <b id="results_title"> Search Similar </b></a>
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

function perform_search(event){
  
  // Cancel the default action, if needed
  // This prevents adding ?year_filter= in the url
  event.preventDefault();
  
  const input_text = myTextArea.value.trim()
  console.log("input_text")
  console.log(input_text)
  
  const input_filter = myDropdown.options[myDropdown.selectedIndex].value
  console.log("input_filter")
  console.log(input_filter)
  
  if (input_text !=''){
    search(input_text, input_filter)
  }
  
}

// Perform search when user click on the button
myButton.addEventListener("click", perform_search)

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

// Search by url query
function search_by_id() {
  
  const textarea_input = myTextArea.value.trim()
  console.log("textarea_input")
  console.log(textarea_input)
  
  const query_parameters = window.location.search
  console.log("query_parameters")
  console.log(query_parameters)
  
  // Use URLSearchParams to properly parse query parameters
  const urlParams = new URLSearchParams(query_parameters)
  const arxiv_id = urlParams.get('arxiv_id')
  console.log("arxiv_id")
  console.log(arxiv_id)
  
  const filter = urlParams.get('filter')
  console.log("filter")
  console.log(filter)
  
  // TODO check https://stackoverflow.com/questions/10691316/javascript-empty-string-comparison
  if (arxiv_id != null) {
    
    // if url also has a filter, then use it
    // http://localhost:8080/?arxiv_id=2401.07215&filter=year==2025 works
    if (filter != null) {
      search(arxiv_id, filter)
    }
    // else use no filter
    else {
      search(arxiv_id, "")
    }
  }
}

// Run on page load
search_by_id()