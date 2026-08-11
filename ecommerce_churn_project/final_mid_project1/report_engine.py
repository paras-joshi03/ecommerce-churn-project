from fpdf import FPDF
import datetime

class ExecutiveReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Early Churn Prediction - Executive Summary', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()} | Generated on {datetime.datetime.now().strftime("%Y-%m-%d")}', 0, 0, 'C')

def create_report(metrics_dict, risk_summary):
    pdf = ExecutiveReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Section 1: Business Impact
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Business Impact & Revenue at Risk", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Total Revenue Currently at High Risk: Rs. {metrics_dict['total_clv_at_risk']}")
    pdf.multi_cell(0, 10, f"Total Customers Scored: {metrics_dict['total_customers']}")
    pdf.ln(5)

    # Section 2: Model Reliability (The Visibility part your guide wants)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Model Confidence (Technical Diagnostics)", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"The system selected the {metrics_dict['best_model']} based on a Recall score of {metrics_dict['recall_score']:.2f}.")
    pdf.multi_cell(0, 10, "This means we are successfully catching the majority of potential churners before they leave.")
    
    pdf.output("Churn_Executive_Report.pdf")
    return "Report Generated Successfully"

