$(document).ready(function() {
  $('.navbar-burger').click(function() {
    $('.navbar-burger').toggleClass('is-active');
    $('.navbar-menu').toggleClass('is-active');
  });

  if (window.bulmaCarousel) {
    bulmaCarousel.attach('.carousel', {
      slidesToScroll: 1,
      slidesToShow: 3,
      loop: true,
      infinite: true,
      autoplay: false,
      autoplaySpeed: 3000
    });
  }

  if (window.bulmaSlider) {
    bulmaSlider.attach();
  }

  $('.interactive-gallery-link').on('click', function(event) {
    if (window.location.protocol === 'file:') {
      event.preventDefault();
      window.open('http://localhost:8000/interactive_page/index.html', '_blank', 'noopener');
    }
  });

  var $lightbox = $('#image-lightbox');
  var $lightboxImage = $('#image-lightbox img');

  function closeLightbox() {
    $lightbox.removeClass('is-active').attr('aria-hidden', 'true');
    $lightboxImage.attr('src', '').attr('alt', '');
  }

  $('#ideas .idea-figure img, .cdf-figure img, .method-figure img, .why-loop-figure img, .detail-figure img, .result-figure img, .case-study-figure img').on('click', function() {
    $lightboxImage.attr('src', $(this).attr('src')).attr('alt', $(this).attr('alt') || 'Expanded figure preview');
    $lightbox.addClass('is-active').attr('aria-hidden', 'false');
  });

  $('.image-lightbox-close').on('click', closeLightbox);

  $lightbox.on('click', function(event) {
    if (event.target === this) {
      closeLightbox();
    }
  });

  $(document).on('keydown', function(event) {
    if (event.key === 'Escape') {
      closeLightbox();
    }
  });
});