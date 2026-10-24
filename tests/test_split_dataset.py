from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from split_dataset import Components,allocate,seq_id

class SplitTests(unittest.TestCase):
 def test_transitive_components_and_disjoint_split(self):
  c=Components('abcd');c.join('a','b');c.join('b','c')
  self.assertEqual(c.find('a'),c.find('c'));self.assertNotEqual(c.find('a'),c.find('d'))
  groups=[['a','b','c']]+[[str(i)] for i in range(20)]
  assignments,counts=allocate(groups,42)
  self.assertEqual(len({assignments[x] for x in 'abc'}),1)
  self.assertEqual(sum(counts.values()),23)
  self.assertEqual(allocate(groups,42)[0],assignments)
 def test_exact_sequence_ids(self):
  self.assertEqual(seq_id('ACD'),seq_id('ACD'));self.assertNotEqual(seq_id('ACD'),seq_id('ACE'))

if __name__=='__main__':unittest.main()
