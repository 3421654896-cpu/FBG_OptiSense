import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import capture_preserved_ch1_raw as preserved
import capture_production_pair_teacher as capture
from analyze_production_pair_teacher import analyze,shape_diagnostics
from test_capture_production_pair_teacher import reference_fixture
from test_stress_multirate import _extension, _native_frame
from frame_telemetry import sentinel_offsets_for_version


class FullmapTeacherCaptureTests(unittest.TestCase):
    def test_reference_warmup_rejects_ungapped_wrong_version_or_sparse(self):
        for period_gap,fw in [(True,0x1001f),(False,0x1001f),(True,0x1001e)]:
            ext=bytearray(_extension(0,tuple(range(45)),bandwidth_discontinuity=period_gap))
            ext[24:28]=fw.to_bytes(4,'big')
            port=mock.Mock()
            port.read.return_value=_native_frame(bytes(ext))
            if period_gap and fw==0x1001f:
                result=preserved.normal_frames(port,1,initial=True,reference_only=True)
                self.assertTrue(result[0]['bandwidth_discontinuity'])
            else:
                with self.assertRaises(ValueError):
                    preserved.normal_frames(port,1,initial=True,reference_only=True)
        indices=tuple(g*5+o for g,o in enumerate(sentinel_offsets_for_version(4)))
        ext=bytearray(_extension(1,indices))
        ext[24:28]=(0x1001f).to_bytes(4,'big')
        port=mock.Mock();port.read.return_value=_native_frame(bytes(ext))
        with self.assertRaises(ValueError):preserved.normal_frames(port,1,reference_only=True)

    def test_same_grating_different_points_are_not_duplicates_but_missing_is_not_complete(self):
        records=[]
        for point in range(5):
            r=capture.decode(reference_fixture(point),point,99,2,4)
            r.update(grating=1,repeat=0)
            records.append(r)
        doc={'schema':'production_pair_same_hold_teacher_v1','trained_model':False,'complete':True,
             'full_map_reference':True,'schedule_versions':[4],'requested_selectors':[2],
             'requested_gratings':[1],'requested_repeats':1,'captures':records,
             'rows':[{'measured_wavelength_nm':1525+p*.02} for p in range(45)],
             'safety_cleanup':{'exact_disarm_ack':True,'exact_soa_shutter_ack':True}}
        self.assertEqual(analyze(doc)['aggregate']['v4_5k']['total'],5)
        self.assertTrue(analyze(doc)['reference_shapes'][0]['complete_five_points'])
        missing=copy.deepcopy(doc); missing['captures'].pop()
        with self.assertRaisesRegex(ValueError,'completeness'):analyze(missing)
        missing['complete']=False
        self.assertFalse(analyze(missing)['source_capture_complete'])

    def test_shape_is_descriptive_and_flat_shapes_are_unresolved(self):
        x=[1525,1525.02,1525.04,1525.07,1525.09]
        teacher=[2,20,50,22,3]
        same=shape_diagnostics(x,teacher,[2*v+10 for v in teacher])
        self.assertAlmostEqual(same['normalized_shape_rmse'],0)
        self.assertAlmostEqual(same['coarse_centroid_shift_pm'],0)
        self.assertGreater(same['rmse_codes'],0)
        self.assertFalse(shape_diagnostics(x,[5]*5,[6]*5)['shape_resolved'])
        with self.assertRaises(ValueError):shape_diagnostics(list(reversed(x)),teacher,teacher)

    def test_existing_output_rejected_before_any_device_open(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'existing.json'; path.write_text('original')
            with mock.patch('sys.argv',['capture','--output',str(path)]),mock.patch.object(capture.serial,'Serial') as device:
                with self.assertRaises(FileExistsError):capture.main()
                device.assert_not_called()
            self.assertEqual(path.read_text(),'original')

    def test_reference_first_frame_failure_disarms_and_shutters(self):
        import json
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'failed.json'
            with mock.patch('sys.argv',['capture','--output',str(path),'--full-map-reference','--gratings','1','--versions','4']), \
                    mock.patch.object(capture,'load_installed_rows',return_value=([],[])), \
                    mock.patch.object(capture.serial,'Serial'), \
                    mock.patch.object(capture,'_safe_disarm',return_value=True) as disarm, \
                    mock.patch.object(capture,'_close_shutter',return_value=True) as shutter, \
                    mock.patch.object(capture,'wait_single',return_value=((2,2),capture.app.ACK_VALUE)), \
                    mock.patch.object(capture,'_wait_multirate_ack'), \
                    mock.patch.object(capture,'normal_frames',side_effect=ValueError('bad reference')):
                with self.assertRaisesRegex(ValueError,'bad reference'):capture.main()
                self.assertGreaterEqual(disarm.call_count,2)
                shutter.assert_called_once()
            report=json.loads(path.read_text())
            self.assertFalse(report['complete']);self.assertFalse(report['training_eligible'])
            self.assertEqual(report['captures'],[])
            self.assertTrue(report['safety_cleanup']['exact_soa_shutter_ack'])


if __name__=='__main__':unittest.main()
