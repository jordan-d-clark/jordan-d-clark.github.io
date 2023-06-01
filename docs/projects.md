---
layout: page
title: Projects
permalink: /projects/
---

# Projects

Here are some of my projects:

{% for project in site.projects %}
* [{{ project.title }}]({{ project.url }})
{% endfor %}