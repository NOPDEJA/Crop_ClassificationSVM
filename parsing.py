import xml.etree.ElementTree as ET


class MetaParser:
    def __init__(self, xml):
        self.xml = xml
        self.tree = None
        self.root = None
        self.parse_xml()  # Parse the XML file when the class is instantiated

    def parse_xml(self):
        try:
            # Parse the XML file
            self.tree = ET.parse(self.xml)
            self.root = self.tree.getroot()
        except FileNotFoundError:
            raise FileNotFoundError(f"XML file '{self.xml}' not found.")

    def get_boa_offset(self):
        if self.root is None:
            raise ValueError("XML file not parsed. Call parse_xml() method first.")

        # Find BOA_Offset value within Image_Data_Info section
        image_data_info = self.root.find('.//Product_Image_Characteristics')
        if image_data_info is not None:
            boa_offset = float(image_data_info.find('BOA_ADD_OFFSET_VALUES_LIST').find(f"BOA_ADD_OFFSET").text)
            return boa_offset
        else:
            raise ValueError("BOA_ADD_OFFSET_VALUES_LIST not found in XML metadata.")

    def get_boa_quantification_value(self):
        if self.root is None:
            raise ValueError("XML file not parsed. Call parse_xml() method first.")

        # Find Quantification_Value (example) within the XML tree
        image_data_info = self.root.find('.//Product_Image_Characteristics')
        if image_data_info is not None:
            quantification_value = float(
                image_data_info.find('QUANTIFICATION_VALUES_LIST').find('BOA_QUANTIFICATION_VALUE').text)
            return quantification_value
        else:
            raise ValueError("QUANTIFICATION_VALUES_LIST not found in XML.")
