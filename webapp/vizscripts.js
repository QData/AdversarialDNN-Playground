#(document).ready(function(){
  $("button").click(function(){
    $.post("handler.php", 
      {
        input : "2"
      },
      function(data, status) {
        alert("Data: " + data + "\nStatus: " + status);
      }
    )
  });
});