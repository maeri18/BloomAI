let clicked = false;
let generate = document.getElementById("generate");
let gif = document.getElementById("gif")
generate.addEventListener("click",function(event){
    paragraph = document.createElement("p");
    paragraph.innerText ="processing...";
    document.body.append(paragraph);
    setTimeout(()=>{paragraph.remove();},2000);

});  

gif.addEventListener("click",function(event){

    paragraph = document.createElement("p");
    paragraph.innerText ="processing...";
    document.body.append(paragraph);
    setTimeout(()=>{paragraph.remove();},2000);

});