import contextlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from rtsp_annotator.ground_litter_source import LatestImageMailbox
from scripts.probe_ground_litter_source import main, summarize_samples, SnapshotWriter
from scripts.guard_ground_litter_source_probe import protection_failure, validate_release


class GroundLitterSourceProbeTests(unittest.TestCase):
    def test_slow_trace_consumer_only_loses_old_telemetry(self):
        mailbox=LatestImageMailbox(mp.get_context('spawn'),4,4,trace_capacity=2)
        mailbox.decoder.value=1
        for i in range(5):mailbox.trace_decoded(i,100+i,.01,1,False,False)
        last,rows,lost=mailbox.trace_after(0)
        self.assertEqual((last,lost),(5,3))
        self.assertEqual([r['source_pts'] for r in rows],[103,104])
        self.assertEqual(mailbox.trace_after(last),(5,[],0))

    def test_trace_contention_does_not_hold_up_reader(self):
        mailbox=LatestImageMailbox(mp.get_context('spawn'),4,4,trace_capacity=2)
        with mailbox.trace_lock:
            mailbox.trace_decoded(1,2,.1,1)
        self.assertEqual(mailbox.diagnostics()['trace_lock_skips'],1)
        self.assertEqual(mailbox.trace_after(0),(0,[],0))

    def test_unknown_opencv_flags_are_null_not_false(self):
        mailbox=LatestImageMailbox(mp.get_context('spawn'),4,4,trace_capacity=2)
        mailbox.decoder.value=2
        mailbox.trace_decoded(1,None,.1,1)
        row=mailbox.trace_after(0)[1][0]
        self.assertIsNone(row['source_pts'])
        self.assertIsNone(row['corrupt'])
        self.assertIsNone(row['keyframe'])
        json.dumps(row,allow_nan=False)

    def test_sampling_measurement_includes_startup_and_avoids_poll_jitter(self):
        rows=[{'arrival_elapsed':5.,'probe_elapsed':5.1,'local_age':.1},
              {'arrival_elapsed':6.,'probe_elapsed':6.9,'local_age':.9}]
        result=summarize_samples(rows,10)
        self.assertEqual(result['observed_mailbox_fps'],.2)
        self.assertEqual(result['interval_max_seconds'],1.)
        self.assertEqual(result['first_frame_seconds'],5.)
        self.assertFalse(result['inference_executed'])
        empty=summarize_samples([],10)
        self.assertEqual(empty['observed_mailbox_fps'],0)
        self.assertIsNone(empty['local_age_p95_seconds'])

    def test_required_pyav_stops_before_starting_source(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules',{'av':None}), patch(
                'scripts.probe_ground_litter_source.read_camera') as reader, patch('sys.argv',[
                    'probe','--device-code','test','--require-pyav','--output',str(Path(tmp)/'run')]):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as exc:
                main()
            self.assertEqual(exc.exception.code,2)
            reader.assert_not_called()
            self.assertFalse((Path(tmp)/'run').exists())

    def test_invalid_source_file_has_no_side_effects_or_secret_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);url_file=root/'url';output=root/'run'
            for value in [None,'https://camera?token=secret','rtsp:///secret',
                          'rtsp://camera:secret/live','rtsp://camera/secret\nrtsp://other/live',
                          'rtsp://camera/live'+' '*16384]:
                if value is not None:url_file.write_text(value)
                stderr=io.StringIO()
                with patch('scripts.probe_ground_litter_source.mp.get_context') as context, patch('sys.argv',[
                        'probe','--source-url-file',str(url_file),'--output',str(output)]):
                    with contextlib.redirect_stderr(stderr),self.assertRaises(SystemExit) as exc:main()
                self.assertEqual(exc.exception.code,2)
                context.assert_not_called()
                self.assertFalse(output.exists())
                self.assertNotIn('secret',stderr.getvalue())

    def test_url_passed_to_reader_but_not_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);url_file=root/'url';output=root/'run'
            url='rtsp://user:secret@camera/live?token=private'
            url_file.write_text(url)
            real_context=mp.get_context('spawn')
            stdout=io.StringIO()
            with patch('scripts.probe_ground_litter_source.mp.get_context',return_value=real_context), patch.object(
                    real_context,'Process') as process_factory, patch('sys.argv',[
                    'probe','--source-url-file',str(url_file),'--output',str(output),'--width','4','--height','4']):
                process=process_factory.return_value
                process.is_alive.return_value=False;process.exitcode=0
                with contextlib.redirect_stdout(stdout):main()
                self.assertEqual(process_factory.call_args.kwargs['args'][-1],url)
                process.start.assert_called_once()
            reports=''.join(p.read_text() for p in output.glob('*.json'))+stdout.getvalue()
            self.assertNotIn('secret',reports)
            self.assertNotIn('private',reports)
            self.assertEqual(json.loads((output/'run.json').read_text())['source_kind'],'authorized_rtsp_url_file')

    def test_snapshot_sampling_is_bounded_and_preserves_frame_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);writer=SnapshotWriter(root,2)
            frame=np.zeros((4,6,3),dtype=np.uint8)
            for i in range(200):writer.save(frame,{'arrival_elapsed':i,'sequence':i+1})
            rows=[json.loads(line) for line in (root/'snapshots.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),60)
            self.assertEqual([r['sequence'] for r in rows],list(range(1,120,2)))
            self.assertEqual(rows[0]['size'],[6,4])
            self.assertEqual(len(list((root/'snapshots').glob('*.jpg'))),60)
            self.assertFalse(rows[0]['osd_time_verified'])

    def test_guard_refuses_unhealthy_unknown_or_changed_production(self):
        base={'containers':[{'id':'prod','running':True}], 'checked_unix':100, 'streams':[{
            'id':'existing','status':'running','metrics':{'pipeline_healthy':True,
            'publish_fps':25,'unique_publish_fps':25,'duplicate_publish_fps':0,'metrics_updated_at_unix':99}}]}
        self.assertIsNone(protection_failure(base,base))
        for key,value in [('pipeline_healthy',False),('publish_fps',.05),
                          ('unique_publish_fps',None),('publish_fps',float('nan')),
                          ('duplicate_publish_fps',1),('metrics_updated_at_unix',0),
                          ('metrics_updated_at_unix',None)]:
            current=json.loads(json.dumps(base))
            current['streams'][0]['metrics'][key]=value
            self.assertIsNotNone(protection_failure(base,current))
        changed=json.loads(json.dumps(base));changed['streams']=[]
        self.assertEqual(protection_failure(base,changed),'production_streams_changed')

    def test_incomplete_release_does_not_launch_mixed_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()
            (path/'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'incomplete_release'):
                validate_release(path)


if __name__=='__main__':unittest.main()
