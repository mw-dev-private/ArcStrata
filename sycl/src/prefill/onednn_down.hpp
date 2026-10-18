// Optional SYCL-only small expert-down experiment. All GPU work uses the caller's in-order queue.
#pragma once
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <stdexcept>
namespace strata::prefill::onednn_down {
inline bool flag(const char* name) { const char* p=std::getenv(name); return p && p[0]=='1'; }
struct Stats {
    std::map<std::tuple<int64_t,int64_t,int64_t>,uint64_t> shapes;
    uint64_t calls=0,checks=0,nonexact=0; double max_abs=0,max_rel=0;
    void report() const {
        if(calls || checks) std::fprintf(stderr,"strata onednn down calls=%llu checks=%llu nonexact=%llu max_abs=%.9g max_rel=%.9g\n",(unsigned long long)calls,(unsigned long long)checks,(unsigned long long)nonexact,max_abs,max_rel);
        for(const auto& [key,n]:shapes) std::fprintf(stderr,"strata gpu f16 shape M=%lld N=%lld K=%lld calls=%llu\n",(long long)std::get<0>(key),(long long)std::get<1>(key),(long long)std::get<2>(key),(unsigned long long)n);
    }
};
inline Stats& statistics() { thread_local Stats s; return s; }
struct Entry {
    dnnl::matmul op;
    dnnl::memory x,w,y;
    std::unordered_map<int,dnnl::memory> args;
    Entry(const dnnl::engine& e,int64_t m,int64_t ldy,const void* X,const void* W,void* Y) {
        using dt=dnnl::memory::data_type;
        auto xd=dnnl::memory::desc({m,640},dt::f16,dnnl::memory::dims{640,1});
        auto wd=dnnl::memory::desc({640,2560},dt::f16,dnnl::memory::dims{1,640});
        auto yd=dnnl::memory::desc({m,2560},dt::f32,dnnl::memory::dims{ldy,1});
        auto pd=dnnl::matmul::primitive_desc(e,xd,wd,yd);
        op=dnnl::matmul(pd);
        x=dnnl::sycl_interop::make_memory(xd,e,dnnl::sycl_interop::memory_kind::usm,const_cast<void*>(X));
        w=dnnl::sycl_interop::make_memory(wd,e,dnnl::sycl_interop::memory_kind::usm,const_cast<void*>(W));
        y=dnnl::sycl_interop::make_memory(yd,e,dnnl::sycl_interop::memory_kind::usm,Y);
        args={{DNNL_ARG_SRC,x},{DNNL_ARG_WEIGHTS,w},{DNNL_ARG_DST,y}};
    }
    void bind(const void* X,const void* W,void* Y) {
        x.set_data_handle(const_cast<void*>(X));w.set_data_handle(const_cast<void*>(W));y.set_data_handle(Y);
    }
};
struct State {
    sycl::queue queue;
    dnnl::engine engine;
    dnnl::stream stream;
    std::map<std::pair<int64_t,int64_t>,std::unique_ptr<Entry>> entries;
    float* reference=nullptr; size_t reference_size=0;
    State(const sycl::queue& q):queue(q),engine(dnnl::sycl_interop::make_engine(queue.get_device(),queue.get_context())),stream(dnnl::sycl_interop::make_stream(engine,queue)) {}
    ~State(){queue.wait();if(reference)sycl::free(reference,queue);}
};
inline bool run(sycl::queue& q,const uint16_t* X,const uint16_t* W,float* Y,int64_t M,int64_t N,int64_t K,int64_t ldy,float beta) {
    static const bool enabled=flag("STRATA_DNN_DOWN");
    static const bool shapes=flag("STRATA_DNN_DOWN_SHAPES");
    static const bool check=flag("STRATA_DNN_DOWN_CHECK");
    if(!enabled && !shapes)return false;
    auto& stats=statistics();
    if(shapes)++stats.shapes[{M,N,K}];
    if(!enabled || N!=2560 || K!=640 || M<1 || M>16 || beta!=0 || ldy!=N)return false;
    if(!q.has_property<sycl::property::queue::in_order>())throw std::runtime_error("oneDNN down requires in-order queue");
    thread_local std::map<sycl::queue*,std::unique_ptr<State>> states;
    auto& ptr=states[&q];if(!ptr)ptr=std::make_unique<State>(q);auto& s=*ptr;
    auto& entry=s.entries[{M,ldy}];if(!entry)entry=std::make_unique<Entry>(s.engine,M,ldy,X,W,Y);
    entry->bind(X,W,Y);entry->op.execute(s.stream,entry->args);++stats.calls;
    if(check) {
        const size_t count=(size_t)M*N;
        if(count>s.reference_size){q.wait();if(s.reference)sycl::free(s.reference,q);s.reference=sycl::malloc_device<float>(count,q);s.reference_size=count;if(!s.reference)throw std::bad_alloc();}
        oneapi::mkl::blas::column_major::gemm(q,oneapi::mkl::transpose::trans,oneapi::mkl::transpose::nontrans,N,M,K,1.f,reinterpret_cast<const sycl::half*>(W),K,reinterpret_cast<const sycl::half*>(X),K,0.f,s.reference,N);
        std::vector<float> got(count),ref(count);q.memcpy(got.data(),Y,count*sizeof(float));q.memcpy(ref.data(),s.reference,count*sizeof(float)).wait();
        bool exact=true;for(size_t i=0;i<count;++i){const double delta=std::abs(double(got[i])-ref[i]);stats.max_abs=std::max(stats.max_abs,delta);stats.max_rel=std::max(stats.max_rel,delta/std::max(1e-6,std::abs(double(ref[i]))));exact=exact && got[i]==ref[i];if(!std::isfinite(got[i])||!std::isfinite(ref[i])||delta>1e-5+1e-5*std::abs(double(ref[i])))throw std::runtime_error("oneDNN down actual-input parity failed");}
        ++stats.checks;if(!exact)++stats.nonexact;
    }
    return true;
}
}
