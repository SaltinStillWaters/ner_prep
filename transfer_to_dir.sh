#! /bin/bash

# Define directory with event files (current directory by default)
DIR="${1:-.}"

# Loop through files in directory
for FILE in "$DIR"/*; do
    if [ -f "$FILE" ]; then
        # Extract the extension (the number after the last dot)
        EXT="${FILE##*.}"

        # Create directory if it doesn't exist
        if [ ! -d "$DIR/$EXT" ]; then
            mkdir "$DIR/$EXT"
        fi

        # Move the file into its directory
        mv "$FILE" "$DIR/$EXT/"
    fi
done

 echo "Files have been successfully reorganized by extension."