//for header

const menuToggle= document.querySelector(".menu-bars");
const nav = document.querySelector("nav ul");


menuToggle.addEventListener("click", () => {
  nav.classList.toggle("slide");
  });

  window.addEventListener('scroll', function() {
    var navbar = document.getElementById('navbar');
    var scrollPosition = window.scrollY;

    // Change background color based on device width
    if (window.innerWidth > 1024) {
        // For larger screens
        if (scrollPosition > 700) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    } else {
        // For mobile devices
        if (scrollPosition > 100) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    }
});
//write code below
