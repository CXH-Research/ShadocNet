from turtle import pd
import qrcode
from fpdf import FPDF
import csv

# Specifying the width and height of A4 Sheet
pdf_w=210
pdf_h=297

# Creating of pdf class to perform operations
class PDF(FPDF):

    # Generating Qr code and saving in a folder.
    # Also adding a box and title to each QR
    def addInfo(self,codeText,s1,s2,w,h, itemCount):
        # self.rect(s1, s2, w,h)
        codeText = 'Code1'
        img=qrcode.make(codeText)
        # Saving each QR code to a qrCodes folder
        img.save("qrCodes/"+codeText+".png")
        # Accessing the saved QR image
        if itemCount == 0:
            # 4.0, 4.0
            # print(s1, s2)
            self.image("qrCodes/Code1.png", x=s1,y=s2,link='', type='', w=20, h=20)
        # elif itemCount == 4:
        #     # 168.0, 4.0
        #     print(s1, s2)
        #     self.image("qrCodes/"+codeText+".png", x=s1+24,y=s2,link='', type='', w=20, h=20)
        # elif itemCount == 30:
        #     # 4.0, 256.0
        #     # print(s1, s2)
        #     self.image("qrCodes/"+codeText+".png", x=s1,y=s2+23,link='', type='', w=20, h=20)
        elif itemCount == 34:
            # 168.0, 256.0
            self.image("qrCodes/Code1.png", x=s1+24,y=s2+23,link='', type='', w=20, h=20)
        # self.set_font('Arial', '', 10)
        # self.text(x=s1+13, y=s2+4, txt=codeText)

    
pdf = PDF()
pdf.set_title("QR Codes")
pdf.add_page()

# All the values of s1,s2,w,h were calculated after trail & error and may vary as per needs

# Reading the codes from csv file
with open('code_list.csv', mode ='r') as file:
    csvFile = csv.reader(file)

    itemCount=0

    # Margin from start of sheet
    s1=1.0

    # Margin from top of sheet
    s2=1.0
    
    # Width of each QR code
    w=38.0
    # w = 20.0

    # Height of each QR code
    h=38.0
    # h = 20.0

    for item in csvFile:
      
        # pdf.addInfo(item[0],s1,s2,w,h)
        if itemCount == 0 or itemCount == 4 or itemCount == 30 or itemCount == 34:
            pdf.addInfo(item[0],s1,s2,w,h, itemCount)
        s1=s1+w+3.0
                
        itemCount+=1
        # As each row can have 5 items
        if(itemCount%5==0):
            s1=1.0
            s2=s2+h+4.0

        if itemCount >= 35 :
            break

        # As each sheet can have 35 items
        if(itemCount%35==0):
            pdf.add_page()
            s1=4.0
            s2=4.0
            w=38.0
            h=38.0
        
        

pdf.output('qr_codes.pdf','F')





