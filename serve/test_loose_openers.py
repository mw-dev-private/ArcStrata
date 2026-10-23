"""Z01: additional declared-tool openers; preserve canonical parser and argument contracts."""
import contextlib,io,itertools,json,threading,unittest
from serve.frontend import OutputParser

OPENERS=['<=f>','<function:f>','<function =f>','<f>']
TOOLS=[{'name':'f','parameters':{'type':'object','properties':{'x':{'type':['string']},'a':{'type':'array'}}}}]
VALUE='before <parameter=demo>inner</parameter>\n</function>\n</tool_call> after'
def wire(op='<function=f>',value='true',closed=True):
    return '<tool_call>'+op+'\n<parameter=x>\n'+value+'\n</parameter>\n<parameter=a>\n[1,2]\n</parameter>\n</function>'+('</tool_call>' if closed else '')
def run(text,width=1,tools=TOOLS,stream=True,recover=False,thinking=False,finish='stop'):
    p=OutputParser(thinking=thinking,tools=tools,stream_tools=stream,recover=recover);e=[]
    for i in range(0,len(text),width):e+=p.feed(text[i:i+width])
    e+=p.finish(finish)
    return e

def calls(e):return [x.call for x in e if x.kind=='tool_call']
def content(e):return ''.join(x.text for x in e if x.kind=='content')

class LooseOpeners(unittest.TestCase):
    def test_new_openers_preserve_complete_and_final_arguments(self):
        for op,width,closed,thinking,recover in itertools.product(OPENERS,[1,2,7,31,9999],[False,True],[False,True],[False,True]):
            with self.subTest(op=op,width=width,closed=closed,thinking=thinking,recover=recover):
                text=('</think>\n' if thinking else '')+wire(op,VALUE,closed)
                e=run(text,width,thinking=thinking,recover=recover);cs=calls(e)
                self.assertEqual([(c.name,c.arguments) for c in cs],[('f',{'x':VALUE,'a':[1,2]})])
                self.assertEqual(content(e),'')
                starts=[x.call for x in e if x.kind=='tool_start'];self.assertEqual(len(starts),1)
                self.assertEqual(starts[0].id,cs[0].id);self.assertEqual(starts[0].arguments,{})
                self.assertEqual(json.loads(''.join(x.text for x in e if x.kind=='tool_args')),{'x':VALUE,'a':[1,2]})

    def test_nonstream_matches_canonical(self):
        for op,width,recover in itertools.product(OPENERS,[1,7,9999],[False,True]):
            cs=calls(run(wire(op),width,stream=False,recover=recover))
            self.assertEqual([(c.name,c.arguments) for c in cs],[('f',{'x':'true','a':[1,2]})])

    def test_unknown_and_no_declared_tools_stay_literal(self):
        for op,width,tools in itertools.product(OPENERS,[1,7,9999],[None,[]]):
            text=wire(op);e=run(text,width,tools=tools)
            self.assertEqual(calls(e),[]);self.assertEqual(content(e),text)
        for op,width in itertools.product(['<=other>','<function:other>','<function =other>','<other>'],[1,7,9999]):
            text=wire(op);e=run(text,width);self.assertEqual(calls(e),[]);self.assertEqual(content(e),text)

    def test_quotes_and_unwrapped_tags_stay_text(self):
        for op,width in itertools.product(OPENERS,[1,7,9999]):
            for text in ['```xml\n'+wire(op)+'\n```','`'+wire(op)+'`',wire(op).replace('<tool_call>','').replace('</tool_call>','')]:
                e=run(text,width);self.assertEqual(calls(e),[]);self.assertEqual(content(e),text)

    def test_newline_inside_drifted_tag_stays_literal(self):
        for op,width in itertools.product(['<function\n:f>','<\nf>','<=\nf>','<f\n>'],[1,7,9999]):
            text=wire(op);e=run(text,width);self.assertEqual(calls(e),[]);self.assertEqual(content(e),text)

    def test_partial_openers_do_not_invent_calls(self):
        for op,reason in itertools.product(['<','<=','<function:','<function =','<f'],['stop','length','cancel','error']):
            text='<tool_call>'+op;e=run(text,1,finish=reason);self.assertEqual(calls(e),[]);self.assertEqual(content(e),text)

    def test_legacy_canonical_unknown_and_no_tools_policy_is_unchanged(self):
        for text,tools in [(wire(),None),(wire('<function=other>'),TOOLS)]:
            cs=calls(run(text,1,tools=tools));self.assertEqual(len(cs),1)

    def test_prefix_tools_and_punctuation(self):
        for name in ['f','foo','f.bar','f-bar','f_bar','f0']:
            tools=[{'name':x,'parameters':{'type':'object','properties':{}}} for x in ['f','foo','f.bar','f-bar','f_bar','f0']]
            text='<tool_call><'+name+'></function></tool_call>'
            for width in [1,3,9999]:self.assertEqual([c.name for c in calls(run(text,width,tools=tools))],[name])

    def test_unknown_partial_names_are_not_held_until_a_closer(self):
        for text in ['<tool_call><functionX'+'z'*1000,'<tool_call><=unknown','<tool_call><function:unknown']:
            p=OutputParser(thinking=False,tools=TOOLS)
            ev=p.feed(text)
            self.assertEqual(content(ev),text)
            self.assertEqual(p.buf,'')

    def test_multiple_calls_and_parser_instances_do_not_leak(self):
        for op in OPENERS:
            text='first\n'+wire(op)+'\nsecond\n'+wire(op)
            e=run(text,1);self.assertEqual([c.arguments for c in calls(e)],[{'x':'true','a':[1,2]}]*2)
            e=run(wire(op),1,tools=None);self.assertEqual(calls(e),[])

class PeerFindings(unittest.TestCase):
    def test_completed_loose_call_does_not_rewrite_unfinished_canonical_successor(self):
        tails=['<tool_call>\n<function=f>',
               '<tool_call>\n<function=f>\n<parameter=x>\nv',
               '<tool_call>\n<function=f>\n<parameter=x>\nv\n</parameter>\n</function>']
        for op,tail,width,recover in itertools.product(OPENERS,tails,[1,7,9999],[False,True]):
            with self.subTest(op=op,tail=tail,width=width,recover=recover):
                e=run(wire(op)+'\n'+tail,width,recover=recover,stream=False,finish='length')
                canonical=run(wire()+'\n'+tail,width,recover=recover,stream=False,finish='length')
                self.assertEqual(content(e),content(canonical))
                self.assertEqual([(c.name,c.arguments) for c in calls(e)],
                                 [(c.name,c.arguments) for c in calls(canonical)])

    def test_unfinished_call_returns_the_original_model_opener(self):
        for op,width,reason in itertools.product(OPENERS,[1,3,9999],['stop','length','cancel']):
            for tail in ['', '\n<parameter=x>\nval', '\n<parameter=x>\nval\n</parameter>']:
                text='<tool_call>\n'+op+tail
                e=run(text,width,stream=False,finish=reason)
                self.assertEqual(calls(e),[])
                self.assertEqual(content(e),text)

    def test_two_functions_in_one_recovery_wrapper_keep_both_calls(self):
        def f(op,n):return op+'\n<parameter=x>\n'+str(n)+'\n</parameter>\n</function>\n'
        for a,b,width,stream in itertools.product(['<function=f>']+OPENERS,['<function=f>']+OPENERS,[1,7,9999],[False,True]):
            text='<tool_call>\n'+f(a,1)+f(b,2)+'</tool_call>'
            e=run(text,width,stream=stream,recover=True)
            self.assertEqual([c.arguments for c in calls(e)],[{'x':'1'},{'x':'2'}],(a,b,width,stream))
            self.assertEqual(content(e),'')

    def test_tabs_are_recognized_and_no_schema_does_not_hold_new_prefixes(self):
        for op in ['<\tf\t>','<=\tf>','<function\t:f>','<function\t=\tf>']:
            for width in [1,7,9999]:
                self.assertEqual([c.arguments for c in calls(run(wire(op),width))],[{'x':'true','a':[1,2]}])
        for tools in [None,[]]:
            for text in ['<tool_call><=','<tool_call><function :','<tool_call><  ']:
                p=OutputParser(thinking=False,tools=tools)
                self.assertEqual(content(p.feed(text)),text)
                self.assertEqual(p.buf,'')

    def test_new_tag_whitespace_limit_is_fragmentation_independent(self):
        for n in [0,1,16,17,100]:
            op='<'+' '*n+'f>'
            for width in [1,7,9999]:
                text=wire(op);e=run(text,width)
                if n<=16:self.assertEqual(len(calls(e)),1)
                else:self.assertEqual(calls(e),[]);self.assertEqual(content(e),text)

    def test_large_whitespace_run_is_not_retained_or_reparsed_as_a_tag(self):
        p=OutputParser(thinking=False,tools=TOOLS);e=p.feed('<tool_call><');peak=len(p.buf)
        for _ in range(32000):
            e+=p.feed(' ');peak=max(peak,len(p.buf))
        e+=p.finish('stop')
        self.assertLessEqual(peak,40)
        self.assertEqual(content(e),'<tool_call><'+' '*32000)
        self.assertEqual(calls(e),[])

class McpHandoff(unittest.TestCase):
    def test_new_openers_reach_fake_hub_once_with_complete_arguments(self):
        from serve import server
        from serve.test_final_arguments import FakeHub,FakeService
        for op,closed in itertools.product(OPENERS,[False,True]):
            hub=FakeHub();ev=run(wire(op,VALUE,closed),1)
            with contextlib.redirect_stdout(io.StringIO()):
                out=list(server.run_with_mcp(FakeService(ev),hub,[],TOOLS,{},[1],False,10,10,{},threading.Event(),{'f'}))
            self.assertEqual(hub.seen,[('f',{'x':VALUE,'a':[1,2]})])
            self.assertEqual(sum(k=='done' for k,x in out),1)

    def test_two_loose_calls_in_one_wrapper_reach_the_hub_in_order(self):
        from serve import server
        from serve.test_final_arguments import FakeHub,FakeService
        for a,b in itertools.product(OPENERS,OPENERS):
            def body(op,n):return op+'\n<parameter=x>\n'+str(n)+'\n</parameter>\n</function>\n'
            text='<tool_call>\n'+body(a,1)+body(b,2)+'</tool_call>'
            hub=FakeHub()
            with contextlib.redirect_stdout(io.StringIO()):
                list(server.run_with_mcp(FakeService(run(text,1,recover=True)),hub,[],TOOLS,{},[1],False,10,10,{},threading.Event(),{'f'}))
            self.assertEqual(hub.seen,[('f',{'x':'1'}),('f',{'x':'2'})])

    def test_length_cancel_and_explicit_cancel_never_execute_fake_tool(self):
        from serve import server
        from serve.test_final_arguments import FakeHub,FakeService
        for op,closed,reason,already_cancelled in itertools.product(OPENERS,[False,True],['stop','length','cancel'],[False,True]):
            if reason=='stop' and not already_cancelled:continue
            hub=FakeHub();cancel=threading.Event()
            if already_cancelled:cancel.set()
            with contextlib.redirect_stdout(io.StringIO()):
                list(server.run_with_mcp(FakeService(run(wire(op,closed=closed),1,finish=reason),reason),hub,[],TOOLS,{},[1],False,10,10,{},cancel,{'f'}))
            self.assertEqual(hub.seen,[])

if __name__=='__main__':unittest.main()
