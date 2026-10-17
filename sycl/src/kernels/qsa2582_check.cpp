// Reuse the repository's deterministic synthetic KV fixture; dump complete outputs, never checksums alone.
#define main scale_bench_unused_main
#include "qsa_attn_scale_bench.cpp"
#undef main
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include "strata/sycl_queue.hpp"
#include <fstream>
namespace strata::kernels { bool qsa_chunk_down_supported(int64_t,int64_t,const QsaShapes&,void*,bool); }
int main(int argc,char** argv){
 if(argc!=2)return 2;
 auto& qu=dpct::get_in_order_queue();
 printf("device=%s generation=%d\n",qu.get_device().get_info<sycl::info::device::name>().c_str(),int(strata::intel_gpu_gen(qu.get_device())));
 std::ofstream dump(argv[1],std::ios::binary);
 for(int cap: {2051,32768}) for(int nq: {1,2,3,4,5,6,32}){
  if(cap==32768&&nq==32)continue;
  Data D; make_pools(D,1,32768,nq);
  sycl::free(D.scratch,qu);D.cap=cap;
  D.scratch=sycl::malloc_device<float>(nq*k::qsa_decode_attn_scratch_floats(cap,D.s),qu);
  std::vector<int32_t> ids(nq*cap),steps(nq*k::kStepCount,0);
  for(int t=0;t<nq;t++){
   steps[t*k::kStepCount+k::kStepWidth]=cap-t*3;
   for(int i=0;i<cap;i++)ids[t*cap+i]=(i*13+t*17)%32768;
  }
  D.d_ids=up(ids);D.d_steps=up(steps);
  // Exercise masked/nonresident pages, shuffled physical layout, and tail chunks.
  std::vector<int32_t> table(8192);qu.memcpy(table.data(),D.pl.page_table,table.size()*4).wait();
  for(int i=0;i<8192;i+=19)table[i]=-1;
  qu.memcpy((void*)D.pl.page_table,table.data(),table.size()*4).wait();
  for(bool prefill: {false,true}) {
  const char* master=std::getenv("STRATA_SYCL_QSA_CHUNK_DOWN");
  const char* phase=std::getenv(prefill ? "STRATA_SYCL_QSA_PREFILL" : "STRATA_SYCL_QSA_DECODE");
  const bool expected=master && master[0]=='1' && (!phase || phase[0]=='1');
  const bool actual=k::qsa_chunk_down_supported(cap,nq,D.s,nullptr,prefill);
  if(actual!=expected){fprintf(stderr,"dispatch mismatch phase%d cap%d T%d\n",prefill,cap,nq);return 4;}
  auto run=[&](){
    if(prefill) k::qsa_prefill_attn_batch(D.d_q,D.pl,D.d_ids,D.d_steps,cap,D.s,D.scratch,D.d_out,nq,nullptr);
    else k::qsa_decode_attn_batch(D.d_q,D.pl,D.d_ids,D.d_steps,cap,D.s,D.scratch,D.d_out,nq,nullptr);
  };
  std::vector<float> output(nq*D.s.n_head*D.s.head_dim);
  run();qu.wait_and_throw();qu.memcpy(output.data(),D.d_out,output.size()*4).wait();
  dump.write((char*)output.data(),output.size()*4);
  namespace sx=sycl::ext::oneapi::experimental;
  sx::command_graph graph{qu.get_context(),qu.get_device(),sycl::property_list{sx::property::graph::enable_native_recording{}}};graph.begin_recording(qu);run();graph.end_recording(qu);
  auto exec=graph.finalize();qu.ext_oneapi_graph(exec).wait_and_throw();
  std::vector<float> got(output.size());qu.memcpy(got.data(),D.d_out,got.size()*4).wait();
  if(memcmp(output.data(),got.data(),got.size()*4)){fprintf(stderr,"graph differs cap%d T%d\n",cap,nq);return 3;}
  printf("phase=%s cap=%d T=%d optimized=%d direct/graph BITWISE EQUAL\n",prefill?"prefill":"decode",cap,nq,actual);fflush(stdout);
  }
  for(const void* p:{(const void*)D.pl.k_q,(const void*)D.pl.v_q,(const void*)D.pl.k_scale,(const void*)D.pl.v_scale,(const void*)D.pl.page_table,(const void*)D.d_q,(const void*)D.d_out,(const void*)D.scratch,(const void*)D.d_ids,(const void*)D.d_steps})sycl::free((void*)p,qu);
 }
 return 0;
}
