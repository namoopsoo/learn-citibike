git log --pretty=oneline -n 1 > fresh/git_hash.txt
rm foo.zip # discard last one
zip   -r foo fresh -i \*.py \*.txt
