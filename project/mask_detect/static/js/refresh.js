{% static %}
function refresh(){
    var pic = document.getElementById('img1');
    var img = {% static "photo/frame.jpg" %};
    pic.src = img;
    window.setTimeout(refresh,10);
}

window.onload=function(){
    refresh();
}