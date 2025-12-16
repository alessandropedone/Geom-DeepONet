#!/usr/bin/env bash
set -e

# Make the user choose if they want to build the docs using also private functions
echo "Do you want to include private functions (those starting with an underscore) in the docs? (y/n)"
read -r INCLUDE_PRIVATE

# If you want to include private members, go to docs/source/conf.py and set autodoc_default_options to include private members

cd docs/source || exit 1
if [ "$INCLUDE_PRIVATE" = "y" ] || [ "$INCLUDE_PRIVATE" = "Y" ]; then
    echo "▶ Including private functions in the docs..."
    sed -i "s/private-members': *False/private-members': True/" conf.py
elif [ "$INCLUDE_PRIVATE" = "n" ] || [ "$INCLUDE_PRIVATE" = "N" ]; then
    echo "▶ Excluding private functions from the docs..."
    sed -i "s/private-members': *True/private-members': False/" conf.py
else
    echo "Invalid input. Please enter 'y' or 'n'."
    exit 1
fi


echo "▶ Building HTML docs..."
cd ..
make html

echo "✔ Docs built"
