import csv

# Specify the input and output file names
input_file = 'madedataset.txt'  # Change this to your text file name
output_file = 'datasetforapp.csv'  # Desired output CSV file name

# Open the input text file and output CSV file
with open(input_file, 'r') as txt_file, open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write the header row (column names)
    writer.writerow(['Question', 'Answer'])
    
    # Read through the text file and split into question and answer
    question = None
    for line in txt_file:
        line = line.strip()  # Remove any leading/trailing whitespaces
        
        if line.startswith('Question:'):
            if question:  # If there's a previous question, write it
                writer.writerow([question, answer])
            
            # Start a new question
            question = line[len('Question:'):].strip()
        elif line.startswith('Answer:'):
            # Extract the answer
            answer = line[len('Answer:'):].strip()
    
    # Don't forget to write the last question-answer pair
    if question:
        writer.writerow([question, answer])

print(f"Converted {input_file} to {output_file}")