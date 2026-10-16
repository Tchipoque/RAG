"""Simplified block manager used to test Python chunking."""

from dataclasses import dataclass, field


@dataclass
class PhysicalBlock:
    block_id: int
    ref_count: int = 0


class BlockAllocator:
    """Hands out physical KV cache blocks and tracks free ones."""

    def __init__(self, num_blocks: int) -> None:
        self.free_blocks = [PhysicalBlock(i) for i in range(num_blocks)]

    def allocate(self) -> PhysicalBlock:
        if not self.free_blocks:
            raise MemoryError("No free KV cache blocks left")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalBlock) -> None:
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)


@dataclass
class BlockTable:
    """Maps logical blocks of a sequence to physical blocks."""

    blocks: list[PhysicalBlock] = field(default_factory=list)

    def append(self, block: PhysicalBlock) -> None:
        self.blocks.append(block)


def swap_out_to_cpu(table: BlockTable, allocator: BlockAllocator) -> None:
    """Preempt a sequence by releasing its GPU blocks."""
    for block in table.blocks:
        allocator.free(block)
    table.blocks.clear()
