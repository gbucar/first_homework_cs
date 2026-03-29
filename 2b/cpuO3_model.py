# Copyright (c) 2022 The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from gem5.isas import ISA
from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor

from m5.objects import RiscvO3CPU
from m5.objects.FuncUnitConfig import *
from m5.objects.BranchPredictor import (
    TournamentBP,
    MultiperspectivePerceptronTAGE64KB,
    LocalBP
)

from m5.objects import *
from m5.objects.BranchPredictor import *
from m5.objects.FUPool import FUPool
from m5.objects.FuncUnit import OpDesc, FUDesc
from m5.objects.FuncUnitConfig import (
    FP_ALU, IntALU, FP_MultDiv, RdWrPort
)

from m5.objects.IQUnit import IQUnit

from m5.objects.BranchPredictor import BranchPredictor, LTAGE




# O3CPUCore extends RiscvO3CPU. RiscvO3CPU is one of gem5's internal models
# the implements an out of order pipeline. Please refer to
#   https://www.gem5.org/documentation/general_docs/cpu_models/O3CPU
# to learn more about O3CPU.

def create_cpu():
    # RiscvO3CPU is the configurable out-of-order CPU model supplied by gem5 for RISCV

    return the_cpu




class O3CPUCore(RiscvO3CPU):
    def __init__(self):
        """
        :param width: sets the width of fetch, decode, rename, issue, wb, and
        commit stages.
        :param rob_size: determine the number of entries in the reorder buffer.
        :param num_int_regs: determines the size of the integer register file.
        :param num_int_regs: determines the size of the vector/floating point
        register file.
        """
        super().__init__()

        # it inherits BaseO3CPU   
        # Constrains stores loads only
        self.cacheStorePorts = 200 
        self.cacheLoadPorts =200
        # ********************************************************
        #  -- CHANGE HERE THE CPU CONFIGURATION PARAMETERS --    *
        # ********************************************************
        # ****************************
        # - STAGES DELAY in clock cycles
        # ****************************
        # Decode to fetch delay
        self.decodeToFetchDelay = 1 
        # Rename to fetch delay
        self.renameToFetchDelay = 1 
        # Issue/Execute/Writeback to fetch delay
        self.iewToFetchDelay = 1
        # Commit to fetch delay
        self.commitToFetchDelay = 1
        # Rename to decode delay
        self.renameToDecodeDelay = 1
        # Issue/Execute/Writeback to decode delay
        self.iewToDecodeDelay = 1
        # Commit to decode delay
        self.commitToDecodeDelay = 1
        # Fetch to decode delay
        self.fetchToDecodeDelay = 1
        # Issue/Execute/Writeback to rename delay
        self.iewToRenameDelay =1
        # Commit to rename delay
        self.commitToRenameDelay = 1
        # Decode to rename delay
        self.decodeToRenameDelay =1
        # Commit to Issue/Execute/Writeback delay
        self.commitToIEWDelay = 1
        # Rename to Issue/Execute/Writeback delay
        self.renameToIEWDelay = 2
        # Issue to execute delay (internal " "to the IEW stage)
        self.issueToExecuteDelay =1 
        # Issue/Execute/Writeback to commit delay
        self.iewToCommitDelay = 1
        # Rename to reorder buffer delay
        self.renameToROBDelay = 1
        self.trapLatency = 13
        self.fetchTrapLatency = 1
        # **************** ************
        # -- BPU SELECTION
        # ****************************
        # predictors from src/cpu/pred/BranchPredictor.py
        self.branchPred = BranchPredictor(conditionalBranchPred=MultiperspectivePerceptronTAGE64KB(numThreads=self.numThreads))
        # ****************************
        # - FETCH STAGE
        # ****************************
        self.fetchWidth = 2 # number of instructions fetched per cycle
        self.fetchBufferSize = 16
        self.fetchQueueSize = 32 
        self.smtNumFetchingThreads = 1

        # ****************************
        # - DECODE STAGE
        # ****************************
        self.decodeWidth = 2
        # possible values "Dynamic", "Partitioned", "Threshold"
        self.smtROBPolicy = "Partitioned"
        self.smtROBThreshold = 100 
        

        # ****************************
        # - RENAME STAGE
        # ****************************
        self.numROBEntries = 128
        self.numPhysIntRegs = 80
        self.numPhysFloatRegs = 64
        self.renameWidth = 2
        self.numRobs = 2000
        self.numPhysVecPredRegs = 32
        # most ISAs don't use condition-code regs, so default is 0
        self.numPhysCCRegs = 0

        # ****************************
        # - DISPATCH/ISSUE STAGE
        # ****************************
        self.dispatchWidth = 2
        self.issueWidth = 2
        
        # ****************************
        # - EXECUTE STAGE
        # ****************************

        # ********************************
        # -- FUNTIONAL UNITS DEFINITION
        # ********************************
        class CPU_IntALU(IntALU):
            opList = [
                OpDesc(opClass="IntAlu", opLat=1, pipelined=False)
            ]
            count = 2

        class CPU_IntMultDiv(FUDesc):
            opList = [
                OpDesc(opClass="IntMult", opLat=3, pipelined=False),
                OpDesc(opClass="IntDiv", opLat=3, pipelined=False)
            ]
            count = 1

        class CPU_FP_ALU(FP_ALU):
            opList = [
                OpDesc(opClass="FloatAdd", opLat=6, pipelined=True),
                OpDesc(opClass="FloatCmp", opLat=6, pipelined=True),
                OpDesc(opClass="FloatCvt", opLat=6, pipelined=True),     
            ]
            count = 1

        class CPU_FP_MultDiv(FP_MultDiv):
            opList = [
                OpDesc(opClass="FloatMult", opLat=6, pipelined=True),
                OpDesc(opClass="FloatMultAcc", opLat=6, pipelined=True),
                OpDesc(opClass="FloatDiv", opLat=12, pipelined=True),
                OpDesc(opClass="FloatSqrt", opLat=12, pipelined=True),
                OpDesc(opClass="FloatMisc", opLat=4, pipelined=True),
            ]
            count = 1


        class CPU_ReadPort(ReadPort):
            opList = [OpDesc(opClass="MemRead", opLat=1), OpDesc(opClass="FloatMemRead", opLat=1)]
            count = 2

        class CPU_WritePort(WritePort):
            opList = [OpDesc(opClass="MemWrite", opLat=1), OpDesc(opClass="FloatMemWrite", opLat=1)]
            count = 2


        
        class CPUFUPool(FUPool):
            FUList = [
                CPU_IntALU(),
                CPU_IntMultDiv(),
                CPU_FP_ALU(),
                CPU_FP_MultDiv(),
                CPU_ReadPort(),
                CPU_WritePort(),
            ]


        self.instQueues = [IQUnit()]
        self.instQueues[0].fuPool = CPUFUPool()

        # ****************************
        # - WRITE/Memory STAGE
        # ****************************
        self.wbWidth = 2
        self.LQEntries = 32
        self.SQEntries = 32
        # Number of places to shift addr before check
        self.LSQDepCheckShift = 4 
        # Should dependency violations be checked for loads & stores or just stores
        self.LSQCheckLoads = True
        # Number of load/store insts before the dep predictor should be invalidated
        self.store_set_clear_period = 250000 
        # Last fetched store table size
        self.LFSTSize = 1024 
        # SMT LSQ Threshold Sharing Parameter
        self.smtLSQThreshold = 100
        # total store ordering 
        self.needsTSO = False

        # ****************************
        # - COMMIT STAGE
        # ****************************
        self.commitWidth = 2
        self.squashWidth = 2
        # Time buffer size for backwards 
        self.backComSize = 10
        # Time buffer size for forward communication
        self.forwardComSize = 10


# Along with BaseCPUCore, CPUStdCore wraps CPUCore to a core compatible
# with gem5's standard library. Please refer to
#   gem5/src/python/gem5/components/processors/base_cpu_core.py
# to learn more about BaseCPUCore.


class O3CPUStdCore(BaseCPUCore):
    def __init__(self):
        """
        :param width: sets the width of fetch, decode, raname, issue, wb, and
        commit stages.
        :param rob_size: determine the number of entries in the reorder buffer.
        :param num_int_regs: determines the size of the integer register file.
        :param num_int_regs: determines the size of the vector/floating point
        register file.
        """
        core = O3CPUCore()
        super().__init__(core, ISA.RISCV)



# O3CPU along with BaseCPUProcessor wraps CPUCore to a processor
# compatible with gem5's standard library. Please refer to
#   gem5/src/python/gem5/components/processors/base_cpu_processor.py
# to learn more about BaseCPUProcessor.


class O3CPU(BaseCPUProcessor):
    def __init__(self):
        """
        :param width: sets the width of fetch, decode, raname, issue, wb, and
        commit stages.
        :param rob_size: determine the number of entries in the reorder buffer.
        :param num_int_regs: determines the size of the integer register file.
        :param num_int_regs: determines the size of the vector/floating point
        register file.
        """
        cores = [O3CPUStdCore()]
        super().__init__(cores)
       

    def get_area_score(self):
        """
        :returns the area score of a pipeline using its parameters width,
        rob_size, num_int_regs, and num_fp_regs.
        """
        score = (
            self._width
            * (2 * self._rob_size + self._num_int_regs + self._num_fp_regs)
            + 4 * self._width
            + 2 * self._rob_size
            + self._num_int_regs
            + self._num_fp_regs
        )
        return score

class RISCV_O3_CPU(O3CPU):
    def __init__(self):
        super().__init__()


