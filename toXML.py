
pic = 3204
#model = 21

categories = ['No_phenotype','Global_sclerosis','Other']
colors = [65280,65535,255]

with open(f'{pic}_pred.xml','w') as fout:

    fout.write('<Annotations MicronsPerPixel="0.252700">\n')
    
    for i,category in enumerate(categories):

        fout.write(f'\t<Annotation Id="{i+1}" Name="" ReadOnly="0" NameReadOnly="0" LineColorReadOnly="0" Incremental="0" Type="4" LineColor="{colors[i]}" Visible="1" Selected="0" MarkupImagePath="" MacroName="">\n')
        fout.write('\t\t<Attributes>\n')
        fout.write(f'\t\t\t<Attribute Name="{category}" Id="0" Value=""/>\n')
        fout.write('\t\t</Attributes>\n')
        fout.write('\t\t<Regions>\n')
        fout.write('\t\t\t<RegionAttributeHeaders>\n')
        fout.write('\t\t\t\t<AttributeHeader Id="9999" Name="Region" ColumnWidth="-1"/>\n')
        fout.write('\t\t\t\t<AttributeHeader Id="9997" Name="Length" ColumnWidth="-1"/>\n')
        fout.write('\t\t\t\t<AttributeHeader Id="9996" Name="Area" ColumnWidth="-1"/>\n')
        fout.write('\t\t\t\t<AttributeHeader Id="9998" Name="Text" ColumnWidth="-1"/>\n')
        fout.write('\t\t\t\t<AttributeHeader Id="1" Name="Description" ColumnWidth="-1"/>\n')
        fout.write('\t\t\t</RegionAttributeHeaders>\n')
        
        #with open(f'detections_{pic}_model{model}.txt') as fin:
        with open(f'detections_{pic}.txt') as fin:
            id = 1
            for line in fin:
                if category in line:
                    data = line.split()[2:6]
                    #data = [0.35*float(d) for d in data]
                    data = [float(d) for d in data]
                    dw = 0.15*(data[2] - data[0])
                    dh = 0.15*(data[3] - data[1])
                    data[0] -= dw
                    data[2] += dw
                    data[1] -= dh
                    data[3] += dh
                    width = data[2] - data[0]
                    height = data[3] - data[1]
                    area = width*height
                    perimeter = 2*(width + height)
                    
                    fout.write(f'\t\t\t<Region Id="{id}" Type="1" Zoom="0.880491" Selected="0" ImageLocation="" ImageFocus="-1" Length="{perimeter}" Area="{area}" LengthMicrons="{perimeter*0.2527}" AreaMicrons="{area*0.2527*0.2527}" Text="" NegativeROA="0" InputRegionId="0" Analyze="1" DisplayId="1">\n')
                    fout.write('\t\t\t\t<Attributes/>\n')
                    fout.write('\t\t\t\t<Vertices>\n')
                    fout.write(f'\t\t\t\t\t<Vertex X="{data[0]}" Y="{data[1]}" Z="0"/>\n')
                    fout.write(f'\t\t\t\t\t<Vertex X="{data[2]}" Y="{data[1]}" Z="0"/>\n')
                    fout.write(f'\t\t\t\t\t<Vertex X="{data[2]}" Y="{data[3]}" Z="0"/>\n')
                    fout.write(f'\t\t\t\t\t<Vertex X="{data[0]}" Y="{data[3]}" Z="0"/>\n')
                    fout.write('\t\t\t\t</Vertices>\n')
                    fout.write('\t\t\t</Region>\n')
                    id += 1

        fout.write('\t\t</Regions>\n')
        fout.write('\t\t<Plots/>\n')
        fout.write('\t</Annotation>\n')
    fout.write('</Annotations>\n')
                               
                    
