function handleDragOver(event) {
  event.preventDefault();
  event.dataTransfer.dropEffect = "copy";
}

function handleDragEnter(event) {
  event.preventDefault();
  document.getElementById("drop_zone").classList.add("hover");
}

function handleDragLeave(event) {
  event.preventDefault();
  document.getElementById("drop_zone").classList.remove("hover");
}

function handleDrop(event) {
  event.preventDefault();
  document.getElementById("drop_zone").classList.remove("hover");

  var files = event.dataTransfer.files;
  document.getElementById("file_input").files = files;
  document.getElementById("upload_form").submit();
}
