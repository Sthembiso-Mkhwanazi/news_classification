import argparse
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def process_file(input_file, output_file):
    """Reads the input file, processes the text, and writes to the output file."""
    try:
        if not os.path.exists(input_file):
            logging.error(f"Input file '{input_file}' not found.")
            return
        
        with open(input_file, 'r', encoding='utf-8') as infile:
            content = infile.read()
        
        # Example processing: Convert text to uppercase
        processed_content = content.upper()
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(processed_content)
        
        logging.info(f"Processing complete! Output saved to '{output_file}'.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process a text file and save results.")
parser.add_argument("input_file", help="Path to the input text file.")
parser.add_argument("output_file", help="Path to save the output file.")

# Parse arguments
args = parser.parse_args()

# Run the processing function
process_file(args.input_file, args.output_file)