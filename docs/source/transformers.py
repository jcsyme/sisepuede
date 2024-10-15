
# try to pre-build?
import sisepuede.transformers.transformers as trf
from sisepuede.manager.sisepuede_examples import SISEPUEDEExamples

# load it and build it
examples = SISEPUEDEExamples()
df_input = examples("input_data_frame")
Transformers = trf.Transformers(
    {},
    df_input = df_input,
)
