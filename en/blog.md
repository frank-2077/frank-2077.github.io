---
layout: default
title: Blog
lang: en
permalink: /en/blog/
alt_url: /blog/
---

<ul class="blog-posts">
  {% assign posts = site.posts | where: "lang", "en" %}
  {% for post in posts %}
  <li>
    <span><i><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%d %b, %Y" }}</time></i></span>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </li>
  {% endfor %}
</ul>
