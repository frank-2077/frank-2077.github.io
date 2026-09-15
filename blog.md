---
layout: default
title: Blog
permalink: /blog/
---

<ul class="blog-posts">
  {% for post in site.posts %}
  <li>
    <span><i><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%d %b, %Y" }}</time></i></span>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </li>
  {% endfor %}
</ul>
