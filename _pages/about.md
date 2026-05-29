---
layout: about
title: Home
permalink: /
subtitle: PhD in Physics

profile:
  align: right
  image: prof_pic.jpg
  image_circular: false # crops the image to make it circular

selected_papers: false # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page

announcements:
  enabled: false # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 5 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: false
  scrollable: false # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts
---

I am a researcher at the University of Minnesota, Twin Cities. My current work focusses on magnetohydrodynamics, numerical analysis and fluvial turbulence. I completed my PhD with [Robert Lysak](https://cse.umn.edu/physics/robert-lysak) at the same institute in 2025, working on numerical simulations of MHD waves in the Earth's magnetosphere. 

I am open to all sorts of collaboration.

---

<div id="contact-wrap" style="max-width: 480px; margin-top: 1rem;">
  <p>Contact me below.</p>
  <form id="contact-form">
    <div style="margin-bottom: 0.75rem;">
      <input type="email" name="email" placeholder="Your email" required
        style="width: 100%; padding: 0.5rem 0.75rem; border: 1px solid #ccc; border-radius: 4px; font-size: 0.95rem;">
    </div>
    <div style="margin-bottom: 0.75rem;">
      <textarea name="message" rows="4" placeholder="Message" required
        style="width: 100%; padding: 0.5rem 0.75rem; border: 1px solid #ccc; border-radius: 4px; font-size: 0.95rem; resize: vertical;"></textarea>
    </div>
    <button type="submit"
      style="padding: 0.4rem 1.2rem; background: var(--global-theme-color); color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 0.95rem;">
      Send
    </button>
  </form>
  <p id="contact-thanks" style="display: none;">Thank you.</p>
</div>

<script>
  document.getElementById('contact-form').addEventListener('submit', function(e) {
    e.preventDefault();
    fetch('https://formspree.io/f/mdabegaa', {
      method: 'POST',
      body: new FormData(this),
      headers: { 'Accept': 'application/json' }
    }).then(function(r) {
      if (r.ok) {
        document.getElementById('contact-form').style.display = 'none';
        document.getElementById('contact-thanks').style.display = 'block';
      }
    });
  });
</script>