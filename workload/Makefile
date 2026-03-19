GEM5_PATH = /d/hpc/projects/FRI/GEM5/gem5_workspace/gem5/
ISA = riscv
GXX = riscv64-linux-gnu-gcc

SOURCES = $(wildcard *.c)
TARGETS = $(SOURCES:.c=.bin)
ASM_TARGETS = $(SOURCES:.c=.s)

all: $(TARGETS) $(ASM_TARGETS)

%.bin: %.c
	$(GXX) -o $@ $< -O1 -static  -lpthread -I$(GEM5_PATH)/include -L$(GEM5_PATH)/util/m5/build/$(ISA)/out -I$(GEM5_PATH)/util/m5/src -lm5 -DGEM5 -lm

%.s: %.c
	$(GXX) -S -o $@ $< -O1  -I$(GEM5_PATH)/include -I$(GEM5_PATH)/util/m5/src -DGEM5

clean:
	rm -f *.bin *.s