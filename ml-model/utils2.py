import pickle
import json
import pandas as pd 
import numpy as np
import pandas as pd
from datetime import datetime
import time
import lime
import lime.lime_tabular
import re


try:
    import config
except:
    pass
class InsuranceDataPrediction2():
    def __init__(self,file_name):
        # self.file_name=file_name
        # print(file_name)
        self.df=pd.read_csv (file_name)
        # df = pd.read_csv(r'{}'.format(file_name))
        # self.df=df
        self.original_df=pd.read_csv(file_name)
        # self.original_df.to_csv('decoded_file.csv')
        self.df.replace({'Yes':1,'No':0},inplace=True)
        self.df['Treatment'].replace({'Medical management':0 ,'Surgical':1},inplace=True)
        self.df['InsuredSex'].replace({'Male':1, 'Female':0},inplace=True)
        self.df['PolicyTypeName'].replace({'Individual Mediclaim':0},inplace=True)
        self.df['ClaimTypeName'].replace({'Cashless':0,'Reimbursement':1},inplace=True)
        self.df['RelationWithClaimant'].replace({'Self':0},inplace=True)
        #     # ClosureOutput column  is ['Fraud - Not Payable' 'No Opinion' 'Genuine' 'Partially Payable']
        # df['Closure_output1'].replace({'Fraud ':1,'Genuine':0},inplace=True)
        # df=pd.get_dummies(df, columns=['Insured_Profession', 'Diagnosis'])
        # df1= df1.drop(['Closure_output1'],axis=1)
        # self.df=df
        
        self.df1 = self.df
        # self.df1.drop(['Closure_output1'],axis=1,inplace=True)
        print("Type of dataframe df1: ", type(self.df1))
        # df1=pd.get_dummies(df1, columns=['Insured_Profession', 'Diagnosis'])

    

# Record start time
        self.start_time = time.time()  
       
    def load_model(self):
        try:
            with open(config.Model_file_path,"rb")as f:
                self.model = pickle.load(f)
        except:
            with open ("rf_model.pkl","rb")as f:
                self.model = pickle.load(f)

        try:
            with open(config.Json_file_path,"r")as f:
                self.project_data = json.load(f)
        except:
            with open("project_data.json","r")as f:
                self.project_data=json.load(f)

        return self.model,self.project_data
    def get_predicted_Fraud_or_geniune(self):
        # print("Type of dataframe: ", type(self.df))
        import pandas as pd
        self.load_model()
        output=[]
        percetage_of_ouput=[]
        reason=[]
        df10 = pd.read_csv('IFD_encoded_poc_file.csv')
        X = df10.drop(['Closure_output1'], axis=1)
        y = df10['Closure_output1']
        
        # Iterate over the rows of the dataframe
        # print(type(self.df1))
        if hasattr(self, 'df1'):
            print("Type of dataframe: ", type(self.df1))
            for index, row in self.df1.iterrows():
                my_array = row.values.tolist()
                col_dict={'PolicyTypeName': 0, 'ClaimTypeName': 1, 'Treatment': 2, 'Age  ': 3, 'InsuredSex': 4, 'RelationWithClaimant': 5, 'RegistrationNo': 6, 'TotalNoOfBeds': 7, 'ClinicalHistory': 8, 'PastHistory': 9, 'IPD findings': 10, 'TreatingDrCertificate': 11, 'HospBillRs_df2': 12, 'PathologyBillRs_df2': 13, 'PharmacyBillRs_df2': 14, 'ClaimedAmount': 15, 'distance_hs_ins_km': 16, 'policy_is_in_first_year': 17, 'no_of_days_Admit': 18, 'IsBlacklisted_hospital': 19, 'isinsured_agentsame': 20, 'isinsured_Drsame': 21, 'isClaimantName_Drsame': 22, 'isClaimantName_insagentsame': 23, 'labbill_is_more_hosbill': 24, 'ClaimedAmount_isroundfig': 25, 'Hassameinsured_MultipleClaims': 26, 'Hassamepolicyno_MultipleClaims': 27, 'sameagent_MultiplePolicies_sameinsured': 28, 'policy_within_60_days': 29, 'date_diff1': 30, 'within_30_days': 31, 'More_claims_from_same_hospital_same Diagnosis': 32, 'More_than_claims_from_same_hospital_different_Diagnosis': 33, 'DifferentClaimantsAndNumbers_with_samefamily': 34, 'pol_within_180_days': 35, 'polend_in_30Days': 36, 'sameagent_sameHOS': 37, 'MultipleClaims_with_samecompny': 38, 'SameInsuredAndHospital': 39, 'Closure_output1': 40, 'Insured_Profession_ employee': 41, 'Insured_Profession_Business': 42, 'Insured_Profession_Employee at private firm': 43, 'Insured_Profession_Hawker': 44, 'Insured_Profession_Housewife': 45, 'Insured_Profession_Lab employee': 46, 'Insured_Profession_Nurse': 47, 'Insured_Profession_RMO': 48, 'Insured_Profession_Retired': 49, 'Insured_Profession_Service': 50, 
                          'Insured_Profession_doctor': 51, 'Insured_Profession_farmer': 52, 'Insured_Profession_govt employee': 53, 'Insured_Profession_lab owner': 54, 'Insured_Profession_pharmcy employee': 55, 'Insured_Profession_private company employee': 56, 'Insured_Profession_security Personnel': 57, 'Insured_Profession_self employed': 58, 'Insured_Profession_shopkeeper': 59, 'Insured_Profession_student': 60, 'Diagnosis_? Low grade glioma': 61, 'Diagnosis_?Dengue': 62, 'Diagnosis_A Case Of Gullian Syndrome With Quadriparesis With Aspiration Pneumonia/Post Ptca/Bronclial Asthenia P Htn+': 63, 'Diagnosis_ACC Hypertension with Chest pain with syncope': 64, 'Diagnosis_ACS CAD': 65, 'Diagnosis_ACUTE GASTROENTERITIS\t': 66, 'Diagnosis_AFI': 67, 'Diagnosis_AFI Hyponatremia': 68, 'Diagnosis_AGE': 69, 'Diagnosis_ALRTI': 70, 'Diagnosis_APD': 71, 'Diagnosis_Abdomen Pain': 72, 'Diagnosis_Abdomen pain': 73, 'Diagnosis_Abdominal and pelvic pain': 74, 'Diagnosis_Abnormal uterine and vaginal bleeding/AUB': 75, 'Diagnosis_Abnormal uterine bleeding': 76, 'Diagnosis_Abscess In Left Knee With Fever': 77, 'Diagnosis_Acc Htn With Lrti': 78, 'Diagnosis_Accelerated Hypertension': 79, 'Diagnosis_Accessory Cavitated Uterine Mass': 80, 'Diagnosis_Accidental Ingestion Of Weedicide': 81, 'Diagnosis_Accidental Injury shoulder reverese arthroplasty': 82, 'Diagnosis_Accidental consumption of harpic solution': 83, 'Diagnosis_Acl Tear': 84, 'Diagnosis_Acl Tear Medial Meniscus Tear': 85, 'Diagnosis_Acs with Nstemi': 86, 'Diagnosis_Acute Abdomen': 87, 
                          'Diagnosis_Acute Appendcitis with peritonitis': 88, 'Diagnosis_Acute Appendicitis': 89, 'Diagnosis_Acute Bronchial Asthma': 90, 'Diagnosis_Acute Bronchitis': 91, 'Diagnosis_Acute Bronchopneumonia': 92, 'Diagnosis_Acute CVA with Hemiplegia with Denovo DM': 93, 'Diagnosis_Acute Cerebrovascular Accident': 94, 'Diagnosis_Acute Colitis Left Lower Zone Consolidation Acute': 95, 'Diagnosis_Acute Colitis with aki': 96, 'Diagnosis_Acute Coronary Syndrome': 97, 'Diagnosis_Acute Febrile Illines': 98, 'Diagnosis_Acute Febrile Illness': 99, 'Diagnosis_Acute Febrile Illness Diarrhoea': 100, 'Diagnosis_Acute Febrile Illness Lower Respiratory Tract Infection Angina': 101, 'Diagnosis_Acute Febrile Illness With Seizure': 102, 'Diagnosis_Acute Febrile illnesS cure Dehydration Enteric Fever Hypotension': 103, 'Diagnosis_Acute Gastrittis': 104, 'Diagnosis_Acute Gastroenteritis': 105, 'Diagnosis_Acute Gastroenteritis With Some Dehydration': 106, 'Diagnosis_Acute Gastroentertis With Severe Dehydration': 107, 'Diagnosis_Acute Ge': 108, 'Diagnosis_Acute Ge Afi Dehydration': 109, 'Diagnosis_Acute Inferior wall Myocardial infraction with cardiogenic shock': 110, 'Diagnosis_Acute LRTI with pneumonitis with fever': 111, 'Diagnosis_Acute Lrti': 112, 'Diagnosis_Acute Nephrotic Syndrome Renal Biopsy': 113, 'Diagnosis_Acute On Chronic Pancreatitis': 114, 'Diagnosis_Acute Pancreatitis': 115, 'Diagnosis_Acute Post Traumatic Medial Capsule Tear And Medial Ulnar Collateral': 116, 'Diagnosis_Acute Sepsis': 117, 'Diagnosis_Acute Severe Anemia': 118, 'Diagnosis_Acute Typhoid Fever Accelerated Hypertension': 119, 'Diagnosis_Acute Viral Fever': 120, 'Diagnosis_Acute Viral Illness With Thrombocytopenia': 121, 'Diagnosis_Acute With Otitis Media': 122, 'Diagnosis_Acute appendicitis': 123, 'Diagnosis_Acute appendicitis with acitis': 124, 'Diagnosis_Acute bronchitis Acute gastroenteritis': 125, 
                          'Diagnosis_Acute chronic pancreatitis': 126, 'Diagnosis_Acute coronary syndrome gastro intestinal bleed chronic kidney disease': 127, 'Diagnosis_Acute coronary syndrome with arrhythmias': 128, 'Diagnosis_Acute febril illness': 129, 'Diagnosis_Acute febrile illness': 130, 'Diagnosis_Acute febrile illness with bronchitis': 131, 'Diagnosis_Acute febrile illness with sepsis': 132, 'Diagnosis_Acute febrile illness with urinary tract infection': 133, 'Diagnosis_Acute fissure in ano': 134, 'Diagnosis_Acute gastroenteritis': 135, 'Diagnosis_Acute gastroenteritis with Severe dehydration with Lower respiratory tract infection': 136, 'Diagnosis_Acute gastroenteritis with dehydration with G4P1A2L1 with 36 weeks': 137, 'Diagnosis_Acute gastroenteritis with weakness': 138, 'Diagnosis_Acute inferior wall MI moderate LV dysfunction with CAD Tripple vessel disease': 139, 'Diagnosis_Acute ischemic stroke': 140, 'Diagnosis_Acute pancreatitis': 141, 'Diagnosis_Acute respiratory tract infection': 142, 'Diagnosis_Acute slip disc with sciatica': 143, 'Diagnosis_Acute stoke with fascio brachial palsy,recurrent TIA, LT MCA block,t2dm with HTN': 144, 'Diagnosis_Acute viral encephalitis': 145, 'Diagnosis_Adeno Cortical Carcinoma': 146, 'Diagnosis_Afi': 147, 'Diagnosis_Afi & Uti': 148, 'Diagnosis_Afi Acute LRTI': 149, 'Diagnosis_Afi Sepsis Cellulitis Left Breast': 150, 'Diagnosis_Age Copd Htn': 151, 'Diagnosis_Age With Dehydration': 152, 'Diagnosis_Aki With Sepsis': 153, 'Diagnosis_Allergic Airway Disease with OSA': 154, 'Diagnosis_Amlapitta': 155, 'Diagnosis_Angina Htn': 156, 'Diagnosis_Anterior Cruciate Ligament tear Meniscal Tear': 157, 'Diagnosis_Anterior Shoulder Instability': 158, 'Diagnosis_Anterior Wall Myocardiac Infarction': 159, 'Diagnosis_Asthma': 160, 'Diagnosis_Atypical B/L Viral Pneumonitis': 161, 'Diagnosis_Atypical Meningioma': 162, 'Diagnosis_Atypical Viral Pneumonia': 163, 
                          'Diagnosis_B L Lower Limb Deep Venous Thrombosis With IVC And': 164, 'Diagnosis_B L Lower Limb Varicose': 165, 'Diagnosis_Balanoposthitis': 166, 'Diagnosis_Bankerts Lesion': 167, 'Diagnosis_Bilateral Pneumonia': 168, 'Diagnosis_Bilateral Pyelonephritis': 169, 'Diagnosis_Bilateral Renal calculi': 170, 'Diagnosis_Bilateral auxillary fibrolipoma': 171, 'Diagnosis_Bilateral pneumonitis with LRTI': 172, 'Diagnosis_Bimalleolar Left ankle fracture': 173, 'Diagnosis_Bl Renal Calculi Obstructive Uropathy Uti Cystitis': 174, 'Diagnosis_Bleeding Hemorrhoids With Fissure In Ano': 175, 'Diagnosis_Blunt Injury Left Upper Quadrant With Splenic Rupture': 176, 'Diagnosis_Blunt Trauma Over Chest And Abdomen': 177, 'Diagnosis_Blunt Trauma To Abdomen': 178, 'Diagnosis_Blunt chest trauma': 179, 'Diagnosis_Both eye lasik': 180, 'Diagnosis_Both knee joint pain': 181, 'Diagnosis_Breast unspecified': 182, 'Diagnosis_Bronchial Asthma': 183, 'Diagnosis_Bronchiolitis': 184, 'Diagnosis_Bronchiolitis Rd': 185, 'Diagnosis_Bronchiolitis With Respiratory Distress': 186, 'Diagnosis_Bronchitis With Bronchiolitis': 187, 'Diagnosis_Bronchopneumonia': 188, 'Diagnosis_Bronchopneumonia Respiratory Failure So Pulsar Erosion': 189, 'Diagnosis_Bronchopneumonia Viral Fever': 190, 'Diagnosis_Bulky Uterus With Fibroids': 191, 'Diagnosis_Burn Injury': 192, 'Diagnosis_CAD': 193, 'Diagnosis_CAD HTN': 194, 'Diagnosis_CAD with uncontrolled Diabetes Mellitus': 195, 'Diagnosis_CKD': 196, 'Diagnosis_COPD': 197, 'Diagnosis_COVID 19': 198, 'Diagnosis_CRHD': 199, 'Diagnosis_Ca Ampula': 200, 'Diagnosis_Ca Breast': 201, 'Diagnosis_Ca Lower Alveolus Status Post Op HPE Squamous Cell': 202, 'Diagnosis_Ca Lung': 203, 'Diagnosis_Ca Nasopharynx': 204, 'Diagnosis_Ca Ovary': 205, 'Diagnosis_Ca Rectum': 206, 'Diagnosis_Ca Stomach': 207, 'Diagnosis_Ca Tongue': 208, 'Diagnosis_Cad Angina': 209, 'Diagnosis_Cad Pulmonary Thombo Embolism': 210, 'Diagnosis_Cad Unstable Angina': 211, 'Diagnosis_Calculus Renal': 212, 'Diagnosis_Calculus of kidney and ureter': 213, 'Diagnosis_Carcinoma Breast': 214, 'Diagnosis_Carcinoma Buccal Mucosa': 215, 'Diagnosis_Carcinoma Of Buccal mucosa With Treasery With Acute Bleeding With Bony inhol': 216, 'Diagnosis_Caronary Artery Disease': 217, 'Diagnosis_Cataract': 218, 'Diagnosis_Cataract In Left Eye': 219, 
                          'Diagnosis_Cataract In Right Eye': 220, 'Diagnosis_Cataract Re': 221, 'Diagnosis_Cellulitis Of Leg': 222, 'Diagnosis_Cerebellar stroke syndrome': 223, 'Diagnosis_Cerebral infarction': 224, 'Diagnosis_Chest Injury': 225, 'Diagnosis_Chest Pain Atypical Chest Pain Dm Ht Cag Normal Coronaries': 226, 'Diagnosis_Chest Pain Under Evaluation K/C/O hypertension': 227, 'Diagnosis_Cholelithiasis': 228, 'Diagnosis_Cholelithiasis And Fissure': 229, 'Diagnosis_Chondromalacia patella': 230, 'Diagnosis_Chronic Calcific Pancreatitis': 231, 'Diagnosis_Chronic Cervical Erosion With Menorrhagia': 232, 'Diagnosis_Chronic Dacryocystitis': 233, 'Diagnosis_Chronic Fissure In Ano': 234, 'Diagnosis_Chronic Portal Vein Thrombosis': 235, 'Diagnosis_Chronic Tonsilitis': 236, 'Diagnosis_Chronic diseases of tonsils and adenoids': 237, 'Diagnosis_Chronic ischaemic heart disease': 238, 'Diagnosis_Chronic kidney disease': 239, 'Diagnosis_Chronic kidney disease, Renal allograft recipient': 240, 'Diagnosis_Chronic obstructive pulmonary disease type 2 diabetes mellitus': 241, 'Diagnosis_Chronic papillary endocervicitis': 242, 'Diagnosis_Ckd': 243, 'Diagnosis_Clavicle Fracture': 244, 'Diagnosis_Cld Thrombocytopenia': 245, 'Diagnosis_Closed Communited Subtronchantric Fracture Left Hip Joint': 246, 'Diagnosis_Closed Right Distak Radius Fracture/T2DM/HTN Orif': 247, 'Diagnosis_Comminuted Both Bone Distal 3rd Leg Fracture Right': 248, 'Diagnosis_Comminuted Lower End Of Radius': 249, 'Diagnosis_Complex Right Ovarian Cyst': 250, 'Diagnosis_Complicated Dengue Hemorrhagic Fever with Mods with ppolyserositis': 251, 'Diagnosis_Congestive heart failure, Viral pneumonitis, Diabetic ketoacidosis': 252, 'Diagnosis_Conjunctival hemorrhage with C3-C4 root compression': 253, 'Diagnosis_Constipation, colonic diverticulosis, fatty liver, diabetes mellitus, hypertension, dyslipidemia, benign prostatic hyperplasia': 254, 'Diagnosis_Conversion disorder/? Auto Immune encephalitis': 255, 'Diagnosis_Copd Lrti With Respiratory Failure': 256, 'Diagnosis_Coronary artery disease': 257, 'Diagnosis_Cough Fever Pneumonia': 258, 'Diagnosis_Covid 19': 259, 'Diagnosis_Covid 19 Pneumonia': 260, 'Diagnosis_Covid 19 with bilateral pneumonia': 261, 
                          'Diagnosis_Covid positve': 262, 'Diagnosis_Crush injury right upper extremity with loss of skin , subcutaneous tissue , muscle and joint structure on lateral aspect of elbow joint area with dermaabraison right arm with left hand injury involving loss of skin , subcutaneous tissue extensor tendon and parts of metacarpal bones with expose elbow joint of right side with heavy contamination of both extremity wound': 263, 'Diagnosis_Cryptococcosis': 264, 'Diagnosis_Cva Left Mca Infarct': 265, 'Diagnosis_Cyst': 266, 'Diagnosis_Cystitis': 267, 'Diagnosis_D7 Compression Fracture': 268, 'Diagnosis_DRTD': 269, 'Diagnosis_DUB': 270, 'Diagnosis_DVT': 271, 'Diagnosis_Dash By 4 Wheeler With Head Injury': 272, 'Diagnosis_Decrease sleep lass intake of food fearful at tim': 273, 'Diagnosis_Deep Abrasion Over Right Thigh With Haematoma With Closed Fracture Proximal Phalanx Right Little Finger With Abrasion Over Right Little Finger': 274, 'Diagnosis_Delivery': 275, 'Diagnosis_Dengu High Fever': 276, 'Diagnosis_Dengue': 277, 'Diagnosis_Dengue Fever': 278, 'Diagnosis_Dengue Fever And Thrombocytopenia': 279, 'Diagnosis_Dengue Fever R': 280, 'Diagnosis_Dengue Fever With Hepatitis': 281, 'Diagnosis_Dengue Fever With Shock': 282, 'Diagnosis_Dengue Fever With Tcp': 283, 'Diagnosis_Dengue Fever With Thrombo': 284, 'Diagnosis_Dengue Fever With thrombocytopenia With Urinary Tract Infection': 285, 'Diagnosis_Dengue Fever with Thrombocytopenia': 286, 'Diagnosis_Dengue High Fever': 287, 'Diagnosis_Dengue Ns1 Positive Thrombocytopenia': 288, 'Diagnosis_Dengue Shock Syndrome': 289, 'Diagnosis_Dengue With Thrombocytoepnia': 290, 'Diagnosis_Dengue With Thrombocytopenia': 291, 'Diagnosis_Dengue fever': 292, 'Diagnosis_Dengue fever With Thrombocytopenia': 293, 'Diagnosis_Dengue fever [classical dengue]': 294, 'Diagnosis_Dengue fever classical dengue': 295, 'Diagnosis_Dengue fever with anaemia with thrombocytopaenia': 296, 'Diagnosis_Dengue fever with dehydration': 297, 'Diagnosis_Dengue fever with hypoglycemic episodes , CAD,HTN': 298, 'Diagnosis_Dengue fever with sepsis': 299, 'Diagnosis_Dengue fever with thrombocytopenia': 300, 'Diagnosis_Dengue hemorrhagic fever': 301, 'Diagnosis_Dengue jaundice': 302, 'Diagnosis_Dengue positive (NS1)': 303, 
                          'Diagnosis_Dengur Fever': 304, 'Diagnosis_Dermato fibrosarcoma protuberance at ant abdominal wall': 305, 'Diagnosis_Deviation of the Nasal Septum': 306, 'Diagnosis_Diabetes insipidus': 307, 'Diagnosis_Diabetic Left Foot With Cellulitis Left Foot': 308, 'Diagnosis_Diarrhoea with hyponetramia': 309, 'Diagnosis_Disc Prolapse': 310, 'Diagnosis_Dislocation and sprain of joints and ligaments of knee': 311, 'Diagnosis_Dislocation sprain and strain of joints and liga': 312, 'Diagnosis_Disorder of continuity of bone': 313, 'Diagnosis_Displaced Fracture tibia': 314, 'Diagnosis_Distal ileal Stricture': 315, 'Diagnosis_Dm/Cld/Aki/Sepsis/Jaundice/Hyponatremia': 316, 'Diagnosis_Drug Induced Anaphylaxis': 317, 'Diagnosis_Duct ectasia left breast': 318, 'Diagnosis_Dyspesia': 319, 'Diagnosis_Dyspnea Under Evaluation Hypothyroidism': 320, 'Diagnosis_Dyspnea with anaemia with hypoalbuminemia with UTI': 321, 'Diagnosis_Endometrial Polyp': 322, 'Diagnosis_Enteric Fever': 323, 'Diagnosis_Enteric Fever Acute Pharyngitis': 324, 'Diagnosis_Enteric Fever With Septicemia': 325, 'Diagnosis_Enteric Fever With Thrombocytopenia': 326, 'Diagnosis_Enteric Fever With Typhoid Fever': 327, 'Diagnosis_Enteric Proxemia Chest Pain': 328, 'Diagnosis_Enteric fever': 329, 'Diagnosis_Enteric fever with UTI': 330, 'Diagnosis_Enteric fever with collitis': 331, 'Diagnosis_Enteric fever with urinary tract infection with emesis': 332, 'Diagnosis_Enteric fever with urinary tract infections with sepsis': 333, 'Diagnosis_Enteric fever with urosepsis': 334, 'Diagnosis_Enteric fever with vertigo with urinary tract infection with thrombocytopenia with viral hepatitis': 335, 'Diagnosis_Enteric fever with vomiting with Urinary tract Infection': 336, 'Diagnosis_Entrocolitis': 337, 'Diagnosis_Essential (primary) hypertension': 338, 'Diagnosis_FEVER': 339, 'Diagnosis_Fatty (change of) liver, not elsewhere classified ;Gastritis and duodenitis': 340, 'Diagnosis_Febrile Illness Viral': 341, 'Diagnosis_Femur Fracture': 342, 'Diagnosis_Femur fracture midshaft': 343, 'Diagnosis_Fever': 344, 'Diagnosis_Fever Cough': 345, 'Diagnosis_Fever Hepatitis': 346, 'Diagnosis_Fever Under Evaluation': 347, 'Diagnosis_Fever With Focus': 348, 'Diagnosis_Fever With Lrti': 349, 'Diagnosis_Fever of other and unknown origin': 350, 'Diagnosis_Fever unspecified': 351, 
                          'Diagnosis_Fever with sepsis with jaundice': 352, 'Diagnosis_Fibroangioma': 353, 'Diagnosis_Fibroid Uterus Total Laparoscopic Hysterectomy': 354, 'Diagnosis_Fissure In Ano': 355, 'Diagnosis_Fracture': 356, 'Diagnosis_Fracture Clavicle Right': 357, 'Diagnosis_Fracture Terminal Phalanx Ring Fingre And Middle P': 358, 'Diagnosis_Fracture Tibia': 359, 'Diagnosis_Fracture distal phalanx': 360, 'Diagnosis_Fracture of TIBIA anf FIBULA': 361, 'Diagnosis_Fracture of forearm': 362, 'Diagnosis_Fracture of right shoulder': 363, 'Diagnosis_Fracture of shoulder and upper arm': 364, 'Diagnosis_Fracture over third radius\t': 365, 'Diagnosis_Fractured Left Neck Femur': 366, "Diagnosis_Full term with Respiratory distress syndrome capute with septisemia erb's paralysis": 367, 'Diagnosis_G3P2L1D1 With 1 Previous Lscs With 36 Weeks Lscs': 368, 'Diagnosis_G4P3L3 with suspected ruptured ectopic pregnancy': 369, 'Diagnosis_GI bleed': 370, 'Diagnosis_GTC': 371, 'Diagnosis_Gallstone disease with umbilical hernia': 372, 'Diagnosis_Gastroenteritis with HTN': 373, 'Diagnosis_Gi Sepsis': 374, 'Diagnosis_Gridrasy (Sciatica) Greevagraham (Cervical spondylosis)': 375, 'Diagnosis_Gt Fracture Right Shoulder': 376, 'Diagnosis_Gullian Barre Syndrome': 377, 'Diagnosis_Haemorrhoids': 378, 'Diagnosis_Hand Surgery': 379, 'Diagnosis_Head Injury': 380, 'Diagnosis_Head Injury With Comminated Comp Fracture Tibia Fibula Left Side': 381, 'Diagnosis_Head Injury With Rt Clavicle': 382, 'Diagnosis_Head injury': 383, 'Diagnosis_Heart Hailure Ischaemic Heart Disease DM': 384, 'Diagnosis_Heart failure': 385, 'Diagnosis_Heavy menstrual bleeding / endometrial polyp': 386, 'Diagnosis_Hemorrhagic dengue fever with thrombocytopenia': 387, 'Diagnosis_Hemorrhoids and perianal venous thrombosis': 388, 'Diagnosis_Hepatitis': 389, 'Diagnosis_Hepatocellular carcinoma': 390, 'Diagnosis_Hepatocellular carcinoma, chronic liver disease': 391, 'Diagnosis_High Grade Fever': 392, 'Diagnosis_Hill Sachs Lesion': 393, 'Diagnosis_Htn': 394, 'Diagnosis_Htn With Cva Gastritis With Septicemia': 395, 'Diagnosis_Humerus Fracture': 396, 'Diagnosis_Humerus fracture': 397, 'Diagnosis_Hyperpyrexia Dehydration Afi': 398, 'Diagnosis_Hypertension Category': 399, 'Diagnosis_Hypertensive Urgency Newly Detected Dm Nasal Blood Ever': 400, 'Diagnosis_ICMP with severe left ventricular dysfunction cardiogenic shock Acute coronary syndrome Anterior wall myocardial infarction': 401, 'Diagnosis_IHD': 402, 'Diagnosis_IHD, IVD, TMI': 403, 'Diagnosis_Ihd': 404, 
                          'Diagnosis_Incisional Hernia': 405, 'Diagnosis_Incisional hernia': 406, 'Diagnosis_Infective Hepatitis': 407, 'Diagnosis_Infective enteritis with dehydration': 408, 'Diagnosis_Inguinal Hernia': 409, 'Diagnosis_Internal Piles': 410, 'Diagnosis_Intestinal obstruction': 411, 'Diagnosis_Intracranial bleed': 412, 'Diagnosis_Iron deficiency anemia, hiatus hernia, pyrexia of unknown origin, ischemic heart disease': 413, 'Diagnosis_Ischemic Stroke': 414, 'Diagnosis_Ischemic heart Diseases/Unstable Angina Cag Three': 415, 'Diagnosis_Jaundice fever': 416, 'Diagnosis_Ketotic hypoglycemia / acute gastritis / acute adenoiditis': 417, 'Diagnosis_Knee Acl Tear': 418, 'Diagnosis_L5 S1 Extented Disc': 419, 'Diagnosis_LRTI': 420, 'Diagnosis_LRTI Dm Htn': 421, 'Diagnosis_LRTI WITH SEPSIS, HYPERTENSION, TYPE2 DM\t': 422, 'Diagnosis_Lateral Condyle Fracture': 423, 'Diagnosis_Le Cataract': 424, 'Diagnosis_Left Distal Radius Intra Articular Fracture': 425, 'Diagnosis_Left Eye Eyelid Laceration With Canaliculus': 426, 'Diagnosis_Left Inferior Pole Patella Fracture Left Depressed 2 Proximal Tibia': 427, 'Diagnosis_Left Massive Pleural Effusion': 428, 'Diagnosis_Left Renal Calculus Sp Left Rgp Test': 429, 'Diagnosis_Left Shoulder Rotator Cuff Tear and Subacromial Bursitis': 430, 'Diagnosis_Left breast abscess': 431, 'Diagnosis_Left chronic suppurative otitis Media': 432, 'Diagnosis_Left femoral distal 1/3rd displaced two part fracture': 433, 'Diagnosis_Left fracture proximal third humerus, lower GI bleed, acute kidney injury with multiple myeloma': 434, 'Diagnosis_Left renal Mass': 435, 'Diagnosis_Left volar Barton fracture with 5 th metacarpal fracture': 436, 'Diagnosis_Low Back Ache': 437, 'Diagnosis_Lower Gi Bleed Anemia Post Exploratory Laparotomy Rt Hemi Colectomy': 438, 'Diagnosis_Lower Lobe Pneumonia With Enteric Fever': 439, 'Diagnosis_Lower Respiratory Tract Infection': 440, 'Diagnosis_Lower Respiratory Tract infection': 441, 'Diagnosis_Lower respiratory tract infection': 442, 'Diagnosis_Lrti Fever': 443, 'Diagnosis_Lrti Septicemia': 444, 'Diagnosis_Lrti With Wheeze': 445, 'Diagnosis_Lrti with Aki': 446, 'Diagnosis_Lrti/Viral Fever': 447, 'Diagnosis_Lt Vul Calculi': 448, 'Diagnosis_Lumbago Trapezitis': 449, 'Diagnosis_Malaria Vivax': 450, 'Diagnosis_Malaria With Typhoid Fever': 451, 'Diagnosis_Malaria vivax with viral fever': 452, 'Diagnosis_Malignant neoplasm of breast': 453, 'Diagnosis_Maternity LSCS': 454, 'Diagnosis_Medial compartment osteoarthritis of right knee': 455, 'Diagnosis_Menorrhagia': 456, 'Diagnosis_Metastatic Cholangiocarcinoma': 457, 'Diagnosis_Migraine': 458, 'Diagnosis_Moderate Systolic Dysfunction': 459, 'Diagnosis_Multiple Facial Fractures Wrist Fracture': 460, 'Diagnosis_Multiple gall bladder calculi': 461, 'Diagnosis_Multiple myleoma Stage 3 RISS': 462, 'Diagnosis_Myesthenia Gravis': 463, 'Diagnosis_NHL': 464, 'Diagnosis_Necrotizing pancreatitis': 465, 'Diagnosis_Non Hodkins Lymphoma': 466, 'Diagnosis_Non Small Cell Lung Cancer': 467, 'Diagnosis_Non alcoholic Pancreatitis': 468, 'Diagnosis_Nonsuppurative otitis media': 469, 'Diagnosis_Normal Delivery': 470, 'Diagnosis_Oa Both Knee': 471, 'Diagnosis_Oa Knee': 472, 'Diagnosis_Osa Extensive Awmi With Dm': 473, 'Diagnosis_Osteo Arthritis And Poly Arthrosis Post Fever': 474, 'Diagnosis_Osteoarthritis': 475, 'Diagnosis_Osteomyelitis of right foot metal tarsal with non healing discharge wound': 476, 'Diagnosis_Other Gestroenteritis And Colitis OF Infections And Unspecified Origin': 477, 'Diagnosis_Other abnormal uterine and vaginal bleeding': 478, 'Diagnosis_Other acute ischemic heart diseases': 479, 'Diagnosis_Other anaemias1': 480, 'Diagnosis_Other diseases of liver': 481, 'Diagnosis_Other disorders of kidney and ureter, not elsewhere classified': 482, 'Diagnosis_Other disorders of urinary System': 483, 'Diagnosis_Other disorders of urinary system': 484, 'Diagnosis_Other gastroenteritis and colitis of infectious and unspecified origin': 485, 'Diagnosis_Other sepsis': 486, 'Diagnosis_Other superficial mycoses': 487, 'Diagnosis_Ovarian Cyst': 488, 'Diagnosis_Ovarian cyst': 489, 'Diagnosis_P/C Appendicitis UTL': 490, 'Diagnosis_P2 L2': 491, 'Diagnosis_PIVD': 492, 'Diagnosis_PUO': 493, 'Diagnosis_Palpitation With Sinus Tachycardia With Panic Atta': 494, 'Diagnosis_Pancytopenia cause aplastic anemia': 495, 'Diagnosis_Patella': 496, 'Diagnosis_Pelvic inflammatory disease with chronic abdomen with pelvic endometriosis with fibroid uterus': 497, 'Diagnosis_Perforated Appendix': 498, 'Diagnosis_Pleural Effusion': 499, 'Diagnosis_Pneumonia': 500, 'Diagnosis_Pneumonia organism unspecified': 501, 'Diagnosis_Pneumonitis': 502, 'Diagnosis_Pneumothorax': 503, 'Diagnosis_Poorly Controlled Type Diabetes Mellitus Hypert': 504, 'Diagnosis_Post Burn Infection Gangrene': 505, 'Diagnosis_Post Circumcision Infection': 506, 'Diagnosis_Post Fall Acute Pid L4-5 S1 Level With Severe Lift Lower Leg Radiclopathy': 507, 'Diagnosis_Postcholecystectomy Liver Cirrhosis': 508, 'Diagnosis_Potts spine Plural Effusion Miliary Tuberclos': 509, 'Diagnosis_Pregnancy': 510, 'Diagnosis_Prolapse intervertebral disc with neurological deficit': 511, 'Diagnosis_Prostate Abscess': 512, 'Diagnosis_Pseudophakia Both Eyes': 513, 'Diagnosis_Pulmonary Thrombo Embolism': 514, 'Diagnosis_Pyrexia With Urosepsis With Rt. Renal Calculus': 515, 'Diagnosis_RTA - Right acetabulum fracture': 516, 'Diagnosis_RTA cerebral concussion injury with body abrassion': 517, 'Diagnosis_RTA with Head Injury with Maxillary bone fracture Swelling over Left eye Zygomatic arch': 518, 'Diagnosis_RTA with head injury': 519, 'Diagnosis_RTA with head injury CLW over forehead': 520, 'Diagnosis_Radius Ulna Fracture': 521, 'Diagnosis_Re Cataract': 522, 'Diagnosis_Recent AWMI DM II LV dysfunction': 523, 'Diagnosis_Renal calculus': 524, 'Diagnosis_Rheumatic Heart Disease Severe Mitral Stenosis Moderate Mitral Regurgitation Moderate Tricuspid Regurgitation Pulmonary Artery Hypertension Normal Left Ventricular Function Atrial Fibrillation Type Ii Diabetes Mellitus Hypothyroidism Hypovitaminosis': 525, 'Diagnosis_Rheumatic heart disease, mitral valve stenosis, mild lv dysfunction, hypertension': 526, 'Diagnosis_Rib Fracture': 527, 'Diagnosis_Right AC Joint Arthropathy; Acute AC joint tendonitis': 528, 'Diagnosis_Right Elbow Dislocation And Humreal Ligamnet Tear And Radius Fracture': 529, 'Diagnosis_Right Eye Cataract': 530, 'Diagnosis_Right FTP sub acute chronic SDH with mass effect and mid line shaft': 531, 'Diagnosis_Right Humerus Fracture': 532, 'Diagnosis_Right Knee Disarticulation with left side below knee': 533, 'Diagnosis_Right Metacarpal fracture': 534, 'Diagnosis_Right Ovarian Endometriotic Cyst': 535, 'Diagnosis_Right Ureteric Calculus With Hudn Urinary Tract Infection': 536, 'Diagnosis_Right hand crush injury': 537, 'Diagnosis_Right tibia posterior malleolus fracture with tibia-fibular ligament tear': 538, 'Diagnosis_Road traffic accident': 539, 'Diagnosis_Rotator Cuff Tear': 540, 'Diagnosis_Rt Hand Crush Injury': 541, 'Diagnosis_Rt Intertrochanteric Femur': 542, 'Diagnosis_Rt Sided Lower Limbs Cellulitis Septic shock': 543, 'Diagnosis_Rta': 544, 'Diagnosis_Rta Csf Rhinorrhea Pneumocephalus Left Maxillary Sinus': 545, 'Diagnosis_Rta Headinjury Fracture Right Lower End Radius': 546, 'Diagnosis_Rta Injury To Pelvis': 547, 'Diagnosis_Rta Polytrauma': 548, 'Diagnosis_Rta with multiple facial bones': 549, 'Diagnosis_Ruptured Appendix': 550, 'Diagnosis_SDH': 551, 'Diagnosis_SLE': 552, 'Diagnosis_STEMI IWMI': 553, 'Diagnosis_Seizure': 554, 'Diagnosis_Seizure With Viral Fever': 555, 'Diagnosis_Seizure disorder, encephalopathy, avascular necrosis of hip': 556, 'Diagnosis_Sepsis': 557, 'Diagnosis_Sepsis Bronchopneumonia': 558, 'Diagnosis_Sepsis Thrombocytopenia Enteric Fever Puo': 559, 'Diagnosis_Sepsis With Septic Shock': 560, 'Diagnosis_Septic Shock': 561, 'Diagnosis_Septicemia': 562, 'Diagnosis_Septicemia with ARF': 563, 'Diagnosis_Septicemia with bronchitis': 564, 'Diagnosis_Septicemia with typhoid fever with viral hepatitis': 565, 'Diagnosis_Septisemia With Uti': 566, 'Diagnosis_Severe Osteoporiosis Dengue Fever': 567, 'Diagnosis_Severe Tbi Multiple Fractures': 568, 'Diagnosis_Severe abdominal pain, spincter of oddi dysfunction with severe gastritis, k/c/o hypertension': 569, 'Diagnosis_Severe dengue hepatitis, herpes simplex hepatitis, acute coronary syndrome, liver cell failure': 570, 'Diagnosis_Sht Acute In Chronic Renal Failure Severe Aortic Stenosis': 571, 'Diagnosis_Snake Bite': 572, 'Diagnosis_Spb Hemorrhoids': 573, 'Diagnosis_Squamous Cell Carcinoma Of Cervix': 574, 'Diagnosis_Stroke': 575, 'Diagnosis_Subtrochanteric Osteotomy Left Hip': 576, 'Diagnosis_Sudden loss of vision RE': 577, 'Diagnosis_Suncondral Fracture of head of femure (R)': 578, 'Diagnosis_Suppurative and unspecified otitis media': 579, 'Diagnosis_Symptomatic Nasal Septum, With Chronic Sinusitis Nasal Polyp': 580, 'Diagnosis_Symptomatic neurosyphilis': 581, 'Diagnosis_Synovial Sarcoma': 582, 'Diagnosis_TIA': 583, 'Diagnosis_Thoracic aortic aneurysm, sepsis with septic shock': 584, 'Diagnosis_Tm Perforation': 585, 'Diagnosis_Tmt Positive For Inducible Ischemia Cag Mild Cad': 586, 'Diagnosis_Torsion Of Testis Left': 587, 'Diagnosis_Triple Vessel Coronary Artery Disease / Angina On exertion / S/P PTCA To LCx /Diabetes Mellitus Type II/ Diabetic Neuropathy': 588, 'Diagnosis_Triple Vessel Disease': 589, 'Diagnosis_Type 2 Compound Medial Malleoli Fracture With': 590, 'Diagnosis_Type 2 diabetes mellitus': 591, 'Diagnosis_Typhoid': 592, 'Diagnosis_Typhoid Fever': 593, 'Diagnosis_Typhoid Fever with Dehydration with Myopia': 594, 'Diagnosis_Typhoid and paratyphoid fevers': 595, 'Diagnosis_Typhoid fever': 596, 'Diagnosis_Typhoid fevers age': 597, 'Diagnosis_Typhoid with Hyperbilirubinemia': 598, 'Diagnosis_Typical Pneumonia': 599, 'Diagnosis_UTI': 600, 'Diagnosis_Ulcerative colitis': 601, 'Diagnosis_Umbilical Hernia Dm And Htn': 602, 'Diagnosis_Uncontrolled DM / ? DKA': 603, 'Diagnosis_Uncontrolled Type 2 Diabetes Mellitus with Proteinuria with Dyslipidemia with Left Groin Thigh Necrotizing Wound': 604, 'Diagnosis_Unspecified acute lower respiratory infection': 605, 'Diagnosis_Unspecified jaundice': 606, 'Diagnosis_Unstable angina': 607, 'Diagnosis_Unstable angina under evaluation EF 60% good LV function': 608, 'Diagnosis_Upper Respiratory Tract Infection Systemic Htn Dyslipidemia': 609, 'Diagnosis_Upper respiratory tract infection with septicemia': 610, 'Diagnosis_Urinary Tract Infection Cholelithiasis': 611, 'Diagnosis_Urinary tract infection with cystitis': 612, 'Diagnosis_Urti Recurrent Pneumonia Ischemic Dcm With Urti': 613, 'Diagnosis_Uterus With subserosal Fibroid With Right Ovarian Cyst': 614, 'Diagnosis_Uti': 615, 'Diagnosis_Uti With Sepsis': 616, 'Diagnosis_Ventral Intradural Extramedullary Lesion Lymphocytic Infiltrate': 617, 'Diagnosis_Vertebral Compression': 618, 'Diagnosis_Vertigo': 619, 'Diagnosis_Viper Bite': 620, 'Diagnosis_Viral Arthritis': 621, 'Diagnosis_Viral Dengue Fever': 622, 'Diagnosis_Viral Exanthematous illness with sepsis with dehydration with pyrexia': 623, 'Diagnosis_Viral Feer With Urti': 624, 'Diagnosis_Viral Fever': 625, 'Diagnosis_Viral Fever Tcp': 626, 'Diagnosis_Viral Fever With Encephalitis': 627, 'Diagnosis_Viral Fever With Respiratory Tract Infection With Uti': 628, 'Diagnosis_Viral Haemorrhagic fever with shock': 629, 'Diagnosis_Viral Illness': 630, 'Diagnosis_Viral Pneumonia': 631, 'Diagnosis_Viral Pneumonitis': 632, 'Diagnosis_Viral Pyrexia': 633, 'Diagnosis_Viral Pyrexia Sepsis Pneumonia': 634, 'Diagnosis_Viral Pyrexia With Clinical Malaria With Uti Secondary Bacterial': 635, 'Diagnosis_Viral Pyrexia With Thrombocytopenia': 636, 'Diagnosis_Viral Pyrexia With Thrombocytopenia With Uti': 637, 'Diagnosis_Viral fever with thrombocytopenia': 638, 'Diagnosis_Viral meningoencephalitis': 639, 'Diagnosis_Viral pyrexia with clinical malaria with urinary tract infection with secondary bacterial infection with dehydration': 640, 'Diagnosis_Warthins tumor': 641, 'Diagnosis_Wheeze associated lower respiratory infection': 642, 'Diagnosis_acid peptic disorder': 643, 'Diagnosis_bilateral lower limb dvt': 644, 'Diagnosis_bmv , RHD, SEVERE MS , DM , HTN , PVD': 645, 'Diagnosis_conjunctivitis': 646, 'Diagnosis_coronary artery disease with CKD on HD': 647, 'Diagnosis_covid 19 positive': 648, 'Diagnosis_dengue fever with lower respiratory tract infection': 649, 'Diagnosis_enteric fever\t': 650, 'Diagnosis_fever': 651, 'Diagnosis_fever dengue\t': 652, 'Diagnosis_fracture femur': 653, 'Diagnosis_gastroenteritis Diabetic ketoacidosis': 654, 'Diagnosis_glass cut Injury wrist': 655, 'Diagnosis_haemorrhoids': 656, 'Diagnosis_head injury with clw over left eyebrow left lower': 657, 'Diagnosis_intermenstrual bleeding': 658, 'Diagnosis_iron deficiency anemia': 659, 'Diagnosis_left eye cataract': 660, 'Diagnosis_left sided cellititu with Abscess': 661, 'Diagnosis_medistinal lymphadenopathy': 662, 'Diagnosis_pancreatitis': 663, 'Diagnosis_pathological femur': 664, 'Diagnosis_pleomorphic Adenoma': 665, 'Diagnosis_pneumonia': 666, 'Diagnosis_post traumatic L5 S1 PIVD': 667, 'Diagnosis_self fall fracture right hip': 668, 'Diagnosis_septicemia': 669, 'Diagnosis_syncope/?LOC': 670, 'Diagnosis_typhoid fever': 671, 'Diagnosis_urti tonsilar hypertrophy': 672, 'Diagnosis_viral pyrexia with LRTI with Thrombocytopenia': 673}
                profession = my_array[-2]
                my_array[-2]='Insured_Profession_' + str(profession)
                Diagnosis = my_array[-1]
                my_array[-1]='Diagnosis_' + str(Diagnosis)
                my_array[-2]=col_dict[my_array[-2]]
                my_array[-1]=col_dict[my_array[-1]]
                # print(col_dict[my_array[-2]])
      
        
                # print(my_array)
                try:
                    arr = np.zeros(673, dtype=int)
                    arr[my_array[-2]] = 1
                    arr[my_array[-1]] = 1
                    new_my_array = my_array[:-2]

                    array1 = np.array(arr)

                    insurance_data_subscibe1 = self.model.predict([array1])[0]

                    if insurance_data_subscibe1 == 1:
                        output.append('Fraud')
                    else:
                        output.append('Genuine')

                    array2 = array1.reshape(1, -1)
                    insurance_data_percentage = self.model.predict_proba(array2)
                    max_probability = np.max(insurance_data_percentage) * 100
                    percetage_of_ouput.append(max_probability)

                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        np.array(X),
                        feature_names=X.columns,
                        verbose=True,
                        mode='classification'
                    )

                    exp = explainer.explain_instance(X.iloc[2], self.model.predict_proba, num_features=7)
                    trigger_list = exp.as_list()
                    my_list = trigger_list
                    # reason.append(my_list)
                    feature_names = [t[0].split()[2] for t in my_list]
                    desired_patterns = ['DifferentClaimantsAndNumbers_with_samefamily',
                    'polend_in_30Days','Diagnosis_Blunt Injury Left Upper Quadrant With Splenic Rupture','Insured_Profession_Business',
                    'pol_within_180_days','Diagnosis_Chronic Portal Vein Thrombosis','Diagnosis_Left Renal Calculus Sp Left Rgp Test',
                    'Diagnosis_Left volar Barton fracture with 5 th metacarpal fracture',
                    'within_30_days','Diagnosis_Pseudophakia Both Eyes','Diagnosis_Post Burn Infection Gangrene']

# Loop through the list of tuples and extract the matching parts
                    extracted_features = []

                    for item in my_list:
                        for pattern in desired_patterns:
                            match = re.search(pattern, item[0])
                            if match:
                                extracted_features.append(match.group())
                   
                    input_string=str(extracted_features)
                    # Remove leading and trailing characters such as brackets and single quotes
                    cleaned_string = input_string.strip("[]'\"")

                    # Replace underscores with spaces and format the string
                    formatted_string = cleaned_string.replace('_', ' ')

                    reason.append(formatted_string)

                except Exception as e:
                    print(f"An error occurred: {e}")


        else:
            print("Error: No dataframe to iterate over.")
        df2=self.original_df
        df2['percetage_of_ouput']=percetage_of_ouput
        df2['output']=output
        df2['reason']=reason

        # end_time = time.time()

        # # Calculate processing time
        # processing_time = end_time - self.start_time
        # print(f"Time taken to process the file: {processing_time:.6f} seconds")
        # Record end time
      
        import datetime
        import pandas as pd

        
        now = datetime.datetime.now()

        df2['date'] = now.strftime("%Y-%m-%d")
        df2['time'] = now.strftime("%H:%M:%S")

        df2['process_date'] = df2['date'] + ' ' + df2['time']


        
        df2.drop(['date','time'], axis=1, inplace=True)
        
        df2.to_csv('output_file.csv',index_label='Claim No.')
        print(df2['process_date'].dtype)

if __name__ == "__main__":
    file_name='input_file.csv'
    insurance_data_subscibe =  InsuranceDataPrediction2(file_name)
    insurance_data_subscibe .get_predicted_Fraud_or_geniune()
    print()
    print(f"your file is downloaded ")
