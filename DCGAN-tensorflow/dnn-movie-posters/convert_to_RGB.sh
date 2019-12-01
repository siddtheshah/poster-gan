#!/usr/bin/env bash
cd dcgan_movies_posters
for f in $(find . -name '*.jpg' | xargs identify -format "%i %[colorspace]\n" | grep -v sRGB | awk '{print $1;}'); do
 echo "$f"
 convert "$f" -colorspace sRGB -type truecolor "$f"
done
