import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus.flowables import Image

class InsuranceDataPrediction3:
    def __init__(self,claim_number):
        self.claim_number=claim_number
        
    def generate_claim_pdf(self):
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv('output_file.csv')

            # Filter the DataFrame to select the row with the matching claim number
            selected_row = df[df['Claim No.'] == self.claim_number]

            # Check if a row with the given claim number was found
            if not selected_row.empty:
                # Create a dictionary from the selected row
                claim_dict = selected_row.iloc[0].to_dict()
                reason_lines = claim_dict['reason'].strip("[]").replace("'", "").split(', ')
                claim_dict['reason'] = '\n'.join(reason_lines)

                # Create a dynamic PDF filename based on the claim number
                output_pdf_filename = f'{self.claim_number}_final_report.pdf'

                # Generate a PDF file
                doc = SimpleDocTemplate(output_pdf_filename, pagesize=letter)
                elements = []

                heading_style = getSampleStyleSheet()['Heading1']
                heading_style.alignment = 1  # Center alignment
                heading = Paragraph("Investigation Claim Report", heading_style)
                elements.append(heading)

                logo_path = 'static/images/mediprobelogo.png'  # Path to the company logo
                logo = Image(logo_path, width=1.4*inch, height=0.4*inch)
                elements.append(logo)
                elements.append(Spacer(1, 12))

                # Create a table style
                style = getSampleStyleSheet()['Normal']
                style.fontName = 'Helvetica'
                style.fontSize = 10

                # Create a list to store table data
                table_data = []
                table_data.append(["Parameters (Headings)", "Case Details"])

                for key, value in claim_dict.items():
                    # Add key and value as separate rows
                    table_data.append([key, value])

                # Create the table directly from the data
                table = Table(table_data)

                # Define a TableStyle for the table
                table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, 1), colors.yellowgreen),  # Header row background color
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ])

                table.setStyle(table_style)

                # Add the table to the elements list
                elements.append(table)

                # Add a spacer
                elements.append(Spacer(1, 12))

                doc.build(elements)

                print(f"PDF file '{output_pdf_filename}' generated successfully.")
            else:
                print(f"No claim found with claim number {self.claim_number}")

        except FileNotFoundError:
            print("CSV file not found.")
        except ValueError:
            print(f"Please enter a valid integer for the claim number {self.claim_number}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        

if __name__ == "__main__":
    claim_number = 10
    insurance_data_subscribe = InsuranceDataPrediction3(claim_number)
    insurance_data_subscribe.generate_claim_pdf()
    print()
    print(claim_number)
    # print(f"Your file is downloaded.")
