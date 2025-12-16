#!/usr/bin/env bash
set -e

# Make the user choose if they want to build the docs using also private functions
echo "Do you want to include private functions (those starting with an underscore) in the docs? (y/n)"
read -r INCLUDE_PRIVATE

# If you want to include private members, go to docs/source/conf.py and set autodoc_default_options to include private members

cd docs/source || exit 1
if [ "$INCLUDE_PRIVATE" = "y" ] || [ "$INCLUDE_PRIVATE" = "Y" ]; then
    echo "▶ Including private functions in the docs..."
    sed -i "s/autodoc_default_options = {}/autodoc_default_options = { 'members': True, 'private-members': True }/" "conf.py"
else
    echo "▶ Excluding private functions from the docs..."
    sed -i "s/autodoc_default_options = { 'members': True, 'private-members': True }/autodoc_default_options = {}/" "conf.py"
fi


echo "▶ Building HTML docs..."
cd ..
make html

echo "✔ Docs built"
