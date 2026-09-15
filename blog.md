---
layout: default
title: 文章
lang: zh
permalink: /blog/
alt_url: /en/blog/
---

<ul class="blog-posts">
  {% assign posts = site.posts | where: "lang", "zh" %}
  {% for post in posts %}
  <li>
    <span><i><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y年%m月%d日" }}</time></i></span>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </li>
  {% endfor %}
</ul>
