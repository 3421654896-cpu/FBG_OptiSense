"""Beta isolation, genuine single ADC protocol, compatibility and t90 UI."""

import binascii
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from PyQt5 import QtWidgets

import beta_profile
import temporary_test_mode as t
from test_temporary_test_mode import reference, seal, frame
from test_temporary_symmetric_seven import selected, unpack
import test_temporary_test_widget as widget_tests
from temporary_test_widget import TemporaryTestWindow, TemporaryWorker
from capture_temporary_half_height_transients import custom_command, decode_custom, transition_analysis
from test_capture_temporary_half_height_transients import response


def single_frame(plan, channel=1, selector=2, sequence=16, fast=True):
    n = len(plan['rows'])
    group = t.points_per_peak_for_rows(plan['rows'])
    request = t.command(plan['rows'], 32, 123, first_delay_us=850,
                        point_delays_us=[850]*n, adc_channel=channel,
                        feedback_selector=selector, single_sample=True, fast_dac_writes=fast)
    crc = binascii.crc32(request[:796])
    feedback = t._feedback_pair(channel, selector if channel < 2 else 0)
    wire = bytearray(46 + n*11)
    wire[:4] = bytes((217, 157, 16, 2 | int(sequence >= 16) | (0 if channel == 1 else channel << 2)))
    for at, value in ((4,123),(8,0x10073),(12,sequence),(16,sequence*100000),
                      (20,n*1300),(24,crc)):
        wire[at:at+4] = value.to_bytes(4, 'big')
    wire[28:34] = bytes((34,n,16,0,*feedback))
    wire[40:40+n] = bytes(range(n))
    for i in range(n):
        at = 40+n+i*10
        value = (60,250,700,1600,700,250,60)[i % group] if group == 7 else (60,700,1600,700,60)[i % group]
        wire[at:at+2] = value.to_bytes(2, 'big')
        wire[at+2:at+6] = (i*1300+400).to_bytes(4, 'big')
        wire[at+6:at+10] = (i*1300+1255).to_bytes(4, 'big')
    return request, crc, seal(wire)


class ProtocolTests(unittest.TestCase):
    def plans(self):
        table, x, y = reference()
        return [t.select_points(table,x,y), selected()[-1]]

    def test_requests_5_and_7_lossless_and_zero_second_wait(self):
        for plan in self.plans():
            for fast in (False, True):
                packet, _, _ = single_frame(plan, fast=fast)
                self.assertEqual(packet[4], 7)
                self.assertEqual(packet[12:16], b'T1S!')
                self.assertEqual(packet[20:22], b'\0\0')
                self.assertEqual(packet[22:24], bytes((t.points_per_peak_for_rows(plan['rows']), int(fast))))
                self.assertFalse(any(packet[800:]))
                for decoded,row in zip(unpack(packet),plan['rows']):
                    self.assertEqual(decoded, [row['index'],*row['codes'],34])

    def test_all_channels_decode_single_values_not_fabricated_pairs(self):
        for plan in self.plans():
            n = len(plan['rows']); group = t.points_per_peak_for_rows(plan['rows'])
            for channel,selector in ((0,3),(1,2),(2,0),(3,0)):
                _,crc,wire = single_frame(plan,channel,selector)
                data = t.decode(wire,123,crc,first_delay_us=850,point_delays_us=[850]*n,
                                point_count=n,adc_channel=channel,feedback_selector=selector,
                                points_per_peak=group,single_sample=True,fast_dac_writes=True)
                self.assertEqual(len(data['records']),n)
                self.assertEqual(data['samples_per_point'],1)
                self.assertEqual(set(data['records'][0]),{'adc_code','write_end_us','sample_end_us'})
                self.assertFalse(t.quality(data)['difference_check_performed'])
                self.assertIsNone(t.quality(data)['pair_difference_large'])
                self.assertFalse(t.quality(data)['stability_verified'])
                self.assertEqual(t.frame_adc_values(data)[0],60)
                buffer = bytearray(b'junk'+wire[:25])
                self.assertFalse(list(t.extract_frames(buffer)))
                buffer.extend(wire[25:]); self.assertEqual(list(t.extract_frames(buffer)),[wire])
                with self.assertRaises(ValueError):
                    t.decode(wire,123,crc,first_delay_us=850,point_count=n,points_per_peak=group)
                damaged = bytearray(wire); damaged[-9] ^= 1
                with self.assertRaises(ValueError):
                    t.decode(damaged,123,crc,point_count=n,single_sample=True)

    def test_bad_current_waits_flags_order_fail_before_io(self):
        plan = selected()[-1]
        for field,limit in enumerate(t.LIMITS):
            rows = json.loads(json.dumps(plan['rows'])); rows[0]['codes'][field]=limit+1
            with self.assertRaises(ValueError): t.command(rows,32,123,single_sample=True)
        for wait in (0,49,851,15025):
            with self.assertRaises(ValueError):
                t.command(plan['rows'],32,123,single_sample=True,point_delays_us=[wait]*63)
        with self.assertRaises(ValueError): t.command(plan['rows'],32,123,fast_dac_writes=True)
        with self.assertRaises(ValueError): t.command(plan['rows'],32,123,single_sample=1)
        rows = list(reversed(plan['rows']))
        with self.assertRaises(ValueError): t.command(rows,32,123,single_sample=True)

    def test_old_packets_and_old_rejection_remain_explicit(self):
        plan=self.plans()[0]
        old=t.command(plan['rows'],32,123,first_delay_us=850)
        self.assertEqual(old[4],3); self.assertEqual(old[20:22],b'\0\x32')
        reject=bytearray(20); reject[:4]=bytes((217,157,10,127))
        reject[4:8]=(123).to_bytes(4,'big'); reject[8:10]=(256).to_bytes(2,'big')
        reject[-2:]=b'\x9d\xd9'
        self.assertEqual(list(t.extract_frames(bytearray(reject))),[bytes(reject)])
        with self.assertRaises(RuntimeError): t.decode(reject,123,0,single_sample=True)

    def test_mocked_complete_stream_and_final_shutter(self):
        import app_JDSU as app
        plan=selected()[-1]; _,crc,_=single_frame(plan)
        frames=[single_frame(plan,sequence=i)[2] for i in range(32)]
        terminal=bytearray(42); terminal[:4]=bytes((217,157,17,0))
        terminal[4:8]=(123).to_bytes(4,'big'); terminal[8:12]=(0x10073).to_bytes(4,'big')
        terminal[12:16]=(32).to_bytes(2,'big')*2; terminal[18:22]=bytes((1,1,0,2))
        terminal[32:36]=crc.to_bytes(4,'big')
        port=MagicMock(timeout=.03); port.write.side_effect=lambda data:len(data)
        port.read.return_value=b''.join(frames)+seal(terminal)
        worker=MagicMock(); worker._feedback_command_and_ack.return_value=True
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(app,'EqualIntervalWorker',return_value=worker), \
             patch('benchmark_stress_realtime._safe_disarm',return_value=True), \
             patch('benchmark_stress_realtime._close_shutter',return_value=True), \
             patch('temporary_test_mode.time.time_ns',return_value=123):
            report=t.execute_device(port,plan,Path(folder)/'capture.json',cycles=32,
                                    first_delay_us=850,point_delays_us=[850]*63,
                                    single_sample=True,fast_dac_writes=True)
        self.assertTrue(report['complete']); self.assertTrue(report['shutter_ack'])
        self.assertEqual(report['spacing_us'],0); self.assertEqual(len(report['frames']),32)
        self.assertEqual(report['samples_per_point'],1)

    def test_beta_diagnostic_path_is_crc_bound_and_echoed(self):
        plan=selected()[-1]; source,target=plan['rows'][:2]
        packet=custom_command(source,target,9,fast_dac_writes=True)
        self.assertEqual(packet[8],13); self.assertEqual(packet[13:16],b'C2B')
        self.assertEqual(int.from_bytes(packet[36:40],'big'),binascii.crc32(packet[:36]))
        wire=bytearray(response(source,target,9)); wire[3]=13
        self.assertTrue(decode_custom(wire,source,target,9,fast_dac_writes=True)['fast_dac_writes'])
        with self.assertRaises(ValueError): decode_custom(wire,source,target,9)

    def test_beta_firmware_feature_is_off_in_formal_build(self):
        root=Path(__file__).resolve().parents[1]
        features=(root/'JDSU/Core/Inc/firmware_features.h').read_text(encoding='utf-8')
        source=(root/'JDSU/Core/Src/ms5614t.c').read_text(encoding='utf-8')
        self.assertIn('#define FBG_FEATURE_BETA_SINGLE_ADC 0',features)
        self.assertIn('if(singleSample && !FBG_FEATURE_BETA_SINGLE_ADC) reject |= 256U',source)
        self.assertIn('if(!singleSample)',source)
        self.assertIn('singleSample ? 11U',source)
        fast=source.split('static HAL_StatusTypeDef PI11210_ApplyTemporaryCodesFast',1)[1].split('#endif',1)[0]
        self.assertEqual(fast.count('PI11210_EnsureReady()'),1)
        for token in ('IDAC5 << 1','data, 4U','PI11210_LimitCode(IDAC5',
                      'PI11210_LimitCode(IDAC6','channelIndex = 2U','channelIndex < 5U',
                      'PI11210_ForceShutter()','PI11210_STATUS_OVR_TEMP','prevDACValid = 0U'):
            self.assertIn(token,fast)
        self.assertLess(fast.index('if(result != HAL_OK)'),fast.index('HAL_GPIO_WritePin'))


class BetaWidgetTests(unittest.TestCase):
    setUpClass=widget_tests.WidgetTests.__dict__['setUpClass']
    setUp=widget_tests.WidgetTests.setUp

    def window(self):
        with patch.dict(os.environ,{'FBG_STUDIO_BETA':'1'}):
            window=TemporaryTestWindow(None)
        self.addCleanup(window.close)
        # These legacy CH1 stream fixtures are not the new CH0 startup choice.
        window.channel_combo.setCurrentIndex(window.channel_combo.findData(1))
        window._load_machine_defaults()
        window._apply_startup_wait_defaults()
        return window

    def test_beta_diagnostics_require_developer_mode_and_default_single_850(self):
        window=self.window()
        self.assertTrue(window.beta)
        self.assertTrue(window.point_diagnostics_row.isHidden())
        self.assertTrue(window.beta_controls_row.isHidden())
        self.assertTrue(window.beta_notice.isHidden())
        window.set_developer_mode(True)
        self.assertFalse(window.point_diagnostics_row.isHidden())
        self.assertFalse(window.beta_controls_row.isHidden())
        self.assertFalse(window.beta_notice.isHidden())
        self.assertFalse(window.advanced_panel.isVisible())
        self.assertTrue(window.beta_single_adc.isChecked())
        self.assertTrue(window.beta_single_adc.isHidden())
        self.assertFalse(window.beta_controls_row.isAncestorOf(window.beta_single_adc))
        self.assertNotIn('Beta 对照', ' '.join(label.text() for label in
                                             window.findChildren(QtWidgets.QLabel)))
        self.assertFalse(window.beta_fast_dac.isChecked())
        self.assertFalse(window.beta_fast_dac.isEnabled())
        self.assertEqual(window.first_delay.value(),850)
        self.assertFalse(window.use_point_waits.isChecked())
        window.beta_single_adc.setChecked(False)
        self.assertFalse(window.beta_fast_dac.isEnabled())
        window.beta_single_adc.setChecked(True)
        self.assertFalse(window.beta_fast_dac.isEnabled())

    def test_requested_ch0_scan_defaults_and_hidden_single_adc_at_startup(self):
        with patch.dict(os.environ,{'FBG_STUDIO_BETA':'1'}):
            window=TemporaryTestWindow(None)
        self.addCleanup(window.close)
        self.assertEqual(window.selected_channel(),0)
        self.assertEqual(window.scan_start_nm.value(),1525.)
        self.assertEqual(window.scan_stop_nm.value(),1555.)
        self.assertEqual(window.equal_interval_nm.value(),.02)
        self.assertEqual(window.equal_interval_settle.value(),.003)
        self.assertTrue(window.beta_single_adc.isChecked())
        self.assertTrue(window.beta_single_adc.isHidden())
        self.assertNotIn('Beta 对照',' '.join(label.text() for label in window.findChildren(QtWidgets.QLabel)))

    def test_startup_overrides_old_scan_preferences_without_rewriting_saved_data(self):
        import temporary_test_widget as module
        payload=dict(schema='finger_stress_scan_preferences_v1',signal_channel=2,scan_start_nm=1530.,
                     scan_stop_nm=1560.,equal_interval_nm=.04,equal_interval_settle_s=.001)
        module.SCAN_PREFERENCES_PATH.write_text(json.dumps(payload),encoding='utf-8')
        before=module.SCAN_PREFERENCES_PATH.read_bytes()
        with patch.dict(os.environ,{'FBG_STUDIO_BETA':'1'}):
            window=TemporaryTestWindow(MagicMock())
        self.addCleanup(window.close)
        self.assertEqual(window.selected_channel(),0)
        self.assertEqual((window.scan_start_nm.value(),window.scan_stop_nm.value()),(1525.,1555.))
        self.assertEqual((window.equal_interval_nm.value(),window.equal_interval_settle.value()),(.02,.003))
        self.assertEqual(module.SCAN_PREFERENCES_PATH.read_bytes(),before)
        self.assertIsNone(window.plan)  # Never relabel another channel's route as CH0.
        self.assertEqual(window.first_delay.value(),850)
        self.assertFalse(window.use_point_waits.isChecked())

    def timing_report(self,window):
        n=window.point_count()
        return dict(complete=True,exact_disarm_ack=True,exact_soa_shutter_ack=True,
                    route=window.plan['rows'],repeats=3,analysis={'transitions':[
                        dict(route_source_order=i-1,route_target_order=i,
                             target_fullband_index=window.plan['rows'][i]['index'],path_update_us=450,
                             channels={'CH1':dict(meaningful_step=True,t90_us=800,
                                                  t90_from_switch_begin_us=1250)}) for i in range(1,n)]})

    def test_beta_90pct_result_not_automatically_applied(self):
        window=self.window(); window.worker=SimpleNamespace(output='t90.json')
        window.switch_timing_ready(self.timing_report(window))
        self.assertEqual(window.point_waits[0],850)
        self.assertEqual(window.point_waits[1],850)  # First 90% is not a settling recommendation.
        self.assertFalse(window.use_point_waits.isChecked())
        window.worker=None
        window.refresh_wait_controls()
        self.assertTrue(window.beta_timing_results.isEnabled())

    def test_incomplete_shutter_or_stale_route_not_used(self):
        for corrupt in ('shutter','route','order'):
            window=self.window(); window.worker=SimpleNamespace(output='t90.json')
            report=self.timing_report(window)
            report['route']=json.loads(json.dumps(report['route']))
            if corrupt=='shutter':report['exact_soa_shutter_ack']=False
            elif corrupt=='route':report['route'][0]['codes'][0]=0
            else:report['analysis']['transitions'][0]['route_target_order']=2
            window.switch_timing_ready(report)
            self.assertIn('结果未采用',window.status.text())
            self.assertFalse(window.beta_timing_results.isEnabled())
            window.worker=None

    def test_result_table_shows_board_dac_and_t90_times(self):
        from PyQt5 import QtWidgets
        window=self.window(); window.worker=SimpleNamespace(output='t90.json')
        window.switch_timing_ready(self.timing_report(window))
        def inspect(dialog):
            table=dialog.findChild(QtWidgets.QTableWidget)
            self.assertEqual(table.rowCount(),45)
            self.assertEqual(table.item(0,4).text(),'首点/回绕未测')
            self.assertEqual(table.item(1,1).text(),'450.0')
            self.assertEqual(table.item(1,2).text(),'800.0')
            self.assertEqual(table.item(1,3).text(),'1250.0')
            return QtWidgets.QDialog.Rejected
        with patch.object(QtWidgets.QDialog,'exec_',inspect):window.show_switch_timing_results()
        window.worker=None

    def test_balanced_candidate_explicit_route_gain_and_dac_binding(self):
        from test_temporary_wait_balance import timing_report
        from temporary_wait_balance import analyze_wait_balance
        window=self.window()
        window.beta_single_adc.setChecked(False)
        report=timing_report(window.plan['rows'])
        report['wait_balance']=analyze_wait_balance(report,[850]*window.point_count())
        old=self.timing_report(window)
        old['wait_balance']=report['wait_balance']
        window.worker=SimpleNamespace(output='t90.json')
        window.switch_timing_ready(old)
        window.worker=None
        window.generate_beta_balanced_waits()
        self.assertEqual(window.point_waits[0],850)
        self.assertEqual(window.point_waits[1],725)
        self.assertFalse(window.use_point_waits.isChecked())
        self.assertIn('须整条路线',window.status.text())
        window.beta_single_adc.setChecked(True)
        window.beta_fast_dac.setChecked(True)
        window.generate_beta_balanced_waits()
        self.assertIn('候选未采用',window.status.text())

    def test_latest_display_final_drain_keeps_terminal_message_and_board_rate(self):
        from temporary_live_display import LatestFrameMailbox
        window=self.window()
        table,x,y=reference()
        window.plan=t.select_points(table,x,y)
        mailbox=LatestFrameMailbox()
        for n in range(16,50):
            data=t.decode(frame(n),123,456)
            data['quality']=t.quality(data)
            data['cycle_start_us']=n*10000
            mailbox.offer(data)
        worker=SimpleNamespace(take_display_frame=mailbox.take,deleteLater=MagicMock())
        window.worker=worker
        window._flush_temporary_display()
        self.assertIn('100.00 Hz',window.status.text())
        data['sequence']=50; data['cycle_start_us']=500000; mailbox.offer(data)
        window.status.setText('停光未确认，请检查板卡')
        window.finished()
        self.assertEqual(window.status.text(),'停光未确认，请检查板卡')
        self.assertEqual(mailbox.displayed,2)
        self.assertFalse(window._temporary_draw_timer.isActive())

    def test_beta_diagnostic_follows_ch0_not_hardcoded_ch1(self):
        window=self.window()
        route=window.plan
        window.channel_combo.setCurrentIndex(0)
        route['signal_channel']=0
        window.plan=route
        window._sync_dynamic_ui()
        self.assertTrue(window.test_switch_timing.isEnabled())
        self.assertIn('CH0',window.test_switch_timing.toolTip())
        report=self.timing_report(window)
        report.update(signal_channel=0,signal_channel_name='CH0',feedback_selector=2,
                      ch1_feedback_selector=0)
        for transition in report['analysis']['transitions']:
            transition['channels']['CH0']=transition['channels']['CH1']
        window.worker=SimpleNamespace(output='ch0.json')
        window.switch_timing_ready(report)
        self.assertEqual(window.plan['switch_90pct_timing']['signal_channel'],0)
        self.assertEqual(window.plan['switch_90pct_timing']['feedback_selector'],2)
        self.assertIn('CH0 90%',window.status.text())
        window.worker=None


class ProfileTests(unittest.TestCase):
    def test_beta_path_explicit_and_formal_identity_unchanged(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertFalse(beta_profile.beta_enabled())
            beta_profile.enable_beta()
            self.assertTrue(beta_profile.beta_enabled())
            self.assertEqual(Path(os.environ['FBG_STUDIO_DATA_DIR']).name,'beta_data')
        with patch.dict(os.environ,{'FBG_STUDIO_DATA_DIR':'C:/custom-beta'},clear=True):
            beta_profile.enable_beta()
            self.assertEqual(os.environ['FBG_STUDIO_DATA_DIR'],'C:/custom-beta')


if __name__=='__main__':
    unittest.main()
