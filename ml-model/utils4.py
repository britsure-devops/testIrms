import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus.flowables import Image

class InsuranceDataPrediction4:
    def __init__(self, claim_number):
        self.claim_number = claim_number
        self.selected_row = None
        self.take_output = None  # Initialize take_output as None

    def generate_claim_pdf(self):
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv('output_file.csv')

            # Filter the DataFrame to select the row with the matching claim number
            self.selected_row = df[df['Claim No.'] == self.claim_number]
            self.take_output = self.selected_row.iloc[0][-4:].to_numpy()
            # print(self.take_output)
            return list(self.take_output)

        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    claim_number = 32
    insurance_data_subscribe = InsuranceDataPrediction4(claim_number)
    selected_row = insurance_data_subscribe.generate_claim_pdf()
    print()
    print(claim_number)
    print(selected_row)  # Print the selected_row
