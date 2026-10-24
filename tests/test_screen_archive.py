import io
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from screen_archive import operations,ligand_metadata,AssemblyAnnotationMismatch
from unittest.mock import patch
from sequence_cif import read_categories


def category(d,prefix,records):
    for key in records[0]:d[prefix+'.'+key]=[r[key] for r in records]


class ScreenTests(unittest.TestCase):
    def test_operator_product(self):
        self.assertEqual(list(operations('(1-2)(3,4)')),[('1','3'),('1','4'),('2','3'),('2','4')])
        with self.assertRaises(ValueError):list(operations('(1)garbage'))

    def test_assembly_ligand_contact_uses_transforms(self):
        d={}
        category(d,'_pdbx_struct_assembly_gen',[
            dict(assembly_id='1',oper_expression='1',asym_id_list='A'),
            dict(assembly_id='1',oper_expression='2',asym_id_list='L')])
        ops=[]
        for id,shift in [('1',0),('2',-8)]:
            r={'id':id}
            for i in range(1,4):
                for j in range(1,4):r[f'matrix[{i}][{j}]']=str(int(i==j))
                r[f'vector[{i}]']=str(shift if i==1 else 0)
            ops.append(r)
        category(d,'_pdbx_struct_oper_list',ops)
        aa=[]
        for asym,entity,seq,comp,x in [('A','1','1','ALA',0),('L','2','.','ZN',10),('U','3','.','MG',0)]:
            aa.append(dict(label_asym_id=asym,label_entity_id=entity,label_seq_id=seq,
                auth_seq_id='5',pdbx_PDB_ins_code='?',label_comp_id=comp,label_atom_id='CA' if asym=='A' else comp,
                label_alt_id='.',occupancy='1',type_symbol='C' if asym=='A' else comp,
                pdbx_PDB_model_num='1',Cartn_x=str(x),Cartn_y='0',Cartn_z='0'))
        category(d,'_atom_site',aa)
        m=ligand_metadata(d,'A',{'1'},4.5)
        self.assertTrue(m['entry_has_nonwater_ligand'])
        self.assertEqual(len(m['entry_ligands']),2)
        self.assertEqual(len(m['ligand_contacts']),1)
        self.assertEqual(m['ligand_contacts'][0]['component_id'],'ZN')
        self.assertEqual(m['ligand_contacts'][0]['contact_residues'][0]['label_seq_id'],1)
        self.assertEqual(m['ligand_contacts'][0]['minimum_distance_A'],2)
        self.assertEqual(m['backbone_atoms'][0]['xyz_A'],[0.,0.,0.])

    def test_annotation_says_monomer_but_two_generated_copies(self):
        d={}
        category(d,'_pdbx_struct_assembly_gen',[dict(assembly_id='1',asym_id_list='A')])
        atoms={'A':[{'xyz':np.zeros(3)}]}
        with patch('screen_archive.selected_atoms',return_value=(atoms,{},'1')), patch('screen_archive.assembly_instances',return_value=[('A',['1'],np.eye(4)),('A',['2'],np.eye(4))]):
            with self.assertRaisesRegex(AssemblyAnnotationMismatch,'copies=2'):
                ligand_metadata(d,'A',{'1'},4.5)

    def test_stream_filter_preserves_metadata_after_atom_table(self):
        source='''data_X
loop_
_atom_site.label_entity_id
_atom_site.label_comp_id
1 ALA
2 HOH
_entry.id X
'''
        d=read_categories(io.StringIO(source),('_atom_site.','_entry.'),lambda a:a['_atom_site.label_entity_id']=='1')
        self.assertEqual(d['_entry.id'],['X'])
        self.assertEqual(d['_atom_site.label_comp_id'],['ALA'])


if __name__=='__main__':unittest.main()
