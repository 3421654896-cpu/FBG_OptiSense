import unittest
from compare_f45_gain_contacts import validate_comparison


class ComparisonValidation(unittest.TestCase):
    def test_depth_opt_in(self):
        a=dict(point=14,depth_mm=.12,ch1_kohm=5,firmware_version=65590,
               table_crc32=123,machine_contact_xyz_mm=[1,2,3])
        b=dict(a,depth_mm=.3)
        with self.assertRaises(ValueError): validate_comparison([a,b])
        validate_comparison([a,b],True)
        for changed in (dict(b,point=13),dict(b,ch1_kohm=20),dict(b,firmware_version=65589),
                        dict(b,table_crc32=124),dict(b,machine_contact_xyz_mm=[1,2,4])):
            with self.assertRaises(ValueError): validate_comparison([a,changed],True)

    def test_original_gain_comparison(self):
        a=dict(point=14,depth_mm=.12,ch1_kohm=5,firmware_version=65590,
               table_crc32=123,machine_contact_xyz_mm=[1,2,3])
        validate_comparison([a,dict(a,ch1_kohm=20)])
        with self.assertRaises(ValueError): validate_comparison([])


if __name__=='__main__': unittest.main()
