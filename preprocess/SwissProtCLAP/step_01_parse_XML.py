import gzip
from lxml import etree


def parse_xml_file(file_name, seq_output_file_name, func_output_file_name, text_output_file_name):
    f_in = gzip.open(file_name, 'r')
    f_seq_out = open(seq_output_file_name, "w")
    f_func_out = open(func_output_file_name, "w")
    f_text_out = open(text_output_file_name, "w")

    doc = etree.iterparse(f_in, events=("start", "end"))
    _, root = next(doc)
    start_tag = None

    cur_uniprot_id, cur_seq, cur_func, cur_text = None, None, [], []
    function_trigger = False
    for event, element in doc:
        if event == "start" and start_tag is None:
            # print("starting", element.tag, element.text)
            start_tag = element.tag
        elif event == "end" and element.tag == start_tag:
            # print("ending", element.tag, element.text)
            # print(cur_uniprot_id, cur_seq, cur_text)
            cur_text = " ".join(cur_text).strip()
            cur_func = " ".join(cur_func).strip()
            if cur_uniprot_id is not None and cur_seq is not None:
                f_seq_out.write("{},{}\n".format(cur_uniprot_id, cur_seq))
                if len(cur_func) > 0:
                    f_func_out.write("{}\n{}\n".format(cur_uniprot_id, cur_func))
                if len(cur_text) > 0:
                    f_text_out.write("{}\n{}\n".format(cur_uniprot_id, cur_text))
            start_tag = None
            cur_uniprot_id, cur_seq, cur_func, cur_text = None, None, [], []
            root.clear()
        else:
            if element.text is None:
                continue
            # print("middle", event, element.tag, element.text, 'attr:', element.attrib)

            if event=="end":  # We only record it when moving to the ending tag.
                if element.tag == r"{http://uniprot.org/uniprot}accession":
                    cur_uniprot_id = element.text.strip()
                elif element.tag == r"{http://uniprot.org/uniprot}sequence":
                    cur_seq = element.text.strip()
                elif element.tag == r"{http://uniprot.org/uniprot}text":
                    temp_text = element.text.strip()
                    cur_text.append(temp_text)
                    # print("text", event, element.tag, temp_text, 'attr:', element.attrib)
                    if function_trigger:
                        cur_func.append(temp_text)
                        # print("function", event, element.tag, temp_text, 'attr:', element.attrib)
            
            if element.tag == r"{http://uniprot.org/uniprot}comment" and element.attrib['type'] == 'function':
                function_trigger = not function_trigger

    """
    <accession>P0C9F0</accession>
    <comment type="function">
        <text evidence="1">Plays a role in virus cell tropism, and may be required for efficient virus replication in macrophages.</text>
    </comment>
    <comment type="similarity">
        <text evidence="2">Belongs to the asfivirus MGF 100 family.</text>
    </comment>
    <sequence length="122" mass="14969" checksum="C5E63C34B941711C" modified="2009-05-05" version="1">MVRLFYNPIKYLFYRRSCKKRLRKALKKLNFYHPPKECCQIYRLLENAPGGTYFITENMTNELIMIAKDPVDKKIKSVKLYLTGNYIKINQHYYINIYMYLMRYNQIYKYPLICFSKYSKIL</sequence>
    """

    return


if __name__ == "__main__":
    file_template = "uniprot_sprot"

    xml_file_name = "{}.xml.gz".format(file_template)
    seq_output_file_name = "data/{}_xml_seq.txt".format(file_template)
    func_output_file_name = "data/{}_xml_func.txt".format(file_template)
    text_output_file_name = "data/{}_xml_text.txt".format(file_template)
    parse_xml_file(xml_file_name, seq_output_file_name, func_output_file_name, text_output_file_name)
