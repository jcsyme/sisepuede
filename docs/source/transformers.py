
# try to pre-build?
import sisepuede.transformers.transformer_kernels as trf
from sisepuede.manager.sisepuede_examples import SISEPUEDEExamples

# load it and build it
examples = SISEPUEDEExamples()
df_input = examples("input_data_frame")
Transformers = trf.TransformerKernels(
    {},
    df_input = df_input,
)
