function parseTimeStringToSeconds(timeString) {
    if (timeString.length === 5) {
        const [minutes, seconds] = timeString.split(':').map(Number);
        return (minutes * 60) + seconds;
    }
    else {
        const [hours, minutes, seconds] = timeString.split(':').map(Number);
        return (hours * 3600) + (minutes * 60) + seconds;
    }
}

let swingdescript = "This image shows the swing (beat-upbeat) ratios extracted from each instrument in this track. The higher the x-axis value, the more that musician 'swings', with the first note in a pair of eighth notes held for longer. Dotted lines show mean ratio."
let feeldescript = "This image shows the relative position of onsets from each instrument in this track. The x-axis shows the position in the bar, and the y-axis time: the graph should be read from left-to-right, bottom-to-top."
let complexdescript = "This image shows the distribution of rhythmic values extracted from each instrument in the track. The x-axis shows the proportion of a bar taken up by each rhythm, with colors corresponding to the equivalent rhythmic value 'bin'."
let interactdescript = "This image shows the adaptation between the musicians in this track, as modeled with linear phase correction. In the left graph, thicker arrows show that one musician adapts to match the beat of another to a greater degree. In the right graph, higher y-axis values show greater adaptation."

function updateImg(imgSrc) {
    document.getElementById('imgbox').src =`${imgSrc}.svg`;
    let descript = document.getElementById('imgdescription');
    if (imgSrc === 'swing') {
        descript.innerHTML = swingdescript;
    }
    else if (imgSrc === 'feel') {
        descript.innerHTML = feeldescript;
    }
    else if (imgSrc === 'complexity') {
        descript.innerHTML = complexdescript;
    }
    else if (imgSrc === 'interaction') {
        descript.innerHTML = interactdescript;
    }
}

function loadJSON(jsSrc) {
    let request = new XMLHttpRequest();
    request.open("GET", jsSrc, false);
    request.send(null)
    return JSON.parse(request.responseText);
}

function getYTLink(jsObj) {
    const ytlink = jsObj['links']['external'][0].replace('watch?v=', 'embed/');
    return ytlink + "?" + 'start=' + parseTimeStringToSeconds(jsObj['timestamps']['start']) + '&end=' + parseTimeStringToSeconds(jsObj['timestamps']['end']);
}

function formatTitle(jsObj) {
    return jsObj['musicians']['pianist']+ ': ' + jsObj['track_name'] + ` (${jsObj['recording_year']})`;
}