// CURRENTLY NOT BEING USED

var socket = io();

socket.on("update_data", function(data) {
    console.log("Received data:", data); // Log received data to console
    document.getElementById("ear_value").innerText = data.EAR;
    document.getElementById("mar_value").innerText = data.MAR;
});

// function to request data from the /data endpoint
function requestData() {
    fetch('/data')
        .then(response => response.json())
        .then(data => {
            console.log("Fetched data:", data); // Log fetched data to console
            document.getElementById("ear_value").innerText = data.EAR;
            document.getElementById("mar_value").innerText = data.MAR;
        })
        .catch(error => console.error('Error fetching data:', error));
}

// 1000 = 1 second
setInterval(requestData, 1000);