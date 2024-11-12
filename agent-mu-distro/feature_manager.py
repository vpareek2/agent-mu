import torch
from typing import Optional, Dict, List, Set, TypeVar, Any
from dataclasses import dataclass
import numpy as np
from numbers import Number

from config import MarketFeatures

@dataclass
class FeatureMetadata:
    shapes: Dict[str, List[int]]
    dtypes: Dict[str, torch.dtype]
    feature_order: List[str]

class DistributedFeatureManager:
    def __init__(self, world_size: int, rank: int, prefetch_size: int = 2):
        self.world_size = world_size
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}')
        self.prefetch_size = prefetch_size

        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self.prefetch_stream = torch.cuda.Stream(device=self.device)

        self.transfer_complete = torch.cuda.Event(enable_timing=False, blocking=False)
        self.features_ready = torch.cuda.Event(enable_timing=False, blocking=False)

        self.features: Optional[MarketFeatures] = None
        self._tensor_cache: Dict[str, torch.Tensor] = {}
        self._pinned_memory_pool: Dict[str, torch.Tensor] = {}
        self._feature_chunks: Dict[str, List[slice]] = {}

        torch.cuda.set_per_process_memory_fraction(0.9, self.device)

    def _create_feature_metadata(self, features: MarketFeatures) -> FeatureMetadata:
        shapes = {
            'ohlvc': list(features.ohlvc.shape),
            'returns': list(features.returns.shape),
            'sma': list(features.sma.shape),
            'ema': list(features.ema.shape),
            'rsi': list(features.rsi.shape),
            'macd': list(features.macd.shape),
            'reddit_sentiment': list(features.reddit_sentiment.shape),
            'news_sentiment': list(features.news_sentiment.shape),
            'combined_sentiment': list(features.combined_sentiment.shape),
            'day': list(features.day.shape),
            'month': list(features.month.shape)
        }

        dtypes = {name: getattr(features, name).dtype for name in shapes.keys()}
        feature_order = list(shapes.keys())

        return FeatureMetadata(
            shapes=shapes,
            dtypes=dtypes,
            feature_order=feature_order
        )

    def _prepare_pinned_buffers(self, metadata: FeatureMetadata) -> None:
        for name in metadata.feature_order:
            if name not in self._pinned_memory_pool:
                shape = metadata.shapes[name]
                dtype = metadata.dtypes[name]
                self._pinned_memory_pool[name] = torch.empty(
                    shape,
                    dtype=dtype,
                    pin_memory=True
                )

    def _chunk_features(self, metadata: FeatureMetadata) -> None:
        for name, shape in metadata.shapes.items():
            chunk_size = shape[0] // self.world_size
            remainder = shape[0] % self.world_size

            chunks = []
            start = 0
            for i in range(self.world_size):
                size = chunk_size + (1 if i < remainder else 0)
                chunks.append(slice(start, start + size))
                start += size

            self._feature_chunks[name] = chunks

    def _broadcast_packed_features(self, features: Optional[MarketFeatures], metadata: FeatureMetadata) -> Dict[str, torch.Tensor]:
        tensors = {}

        with torch.cuda.stream(self.transfer_stream):
            if self.rank == 0 and features is not None:
                for name in metadata.feature_order:
                    pinned_buffer = self._pinned_memory_pool[name]
                    source_tensor = getattr(features, name)

                    pinned_buffer.copy_(source_tensor)

                    gpu_tensor = pinned_buffer.to(
                        self.device,
                        non_blocking=True
                    )
                    tensors[name] = gpu_tensor
            else:
                for name in metadata.feature_order:
                    tensors[name] = torch.empty(
                        metadata.shapes[name],
                        dtype=metadata.dtypes[name],
                        device=self.device
                    )

            for name in metadata.feature_order:
                torch.distributed.broadcast(tensors[name], src=0)

            self.transfer_complete.record(self.transfer_stream)

        return tensors

    def distribute_features(self, features: Optional[MarketFeatures] = None) -> MarketFeatures:
        """Main method to distribute features across GPUs."""
        try:
            current_stream = torch.cuda.current_stream()
            self.transfer_stream.wait_stream(current_stream)

            metadata = None
            if self.rank == 0:
                if features is None:
                    raise ValueError("Rank 0 must provide features")
                metadata = self._create_feature_metadata(features)

            if self.rank == 0:
                metadata_tensor = torch.tensor([
                    len(metadata.shapes),
                    max(len(shape) for shape in metadata.shapes.values())
                ], dtype=torch.long)
            else:
                metadata_tensor = torch.zeros(2, dtype=torch.long)
            torch.distributed.broadcast(metadata_tensor, src=0)

            if metadata:
                self._prepare_pinned_buffers(metadata)
                self._chunk_features(metadata)

            tensors = self._broadcast_packed_features(features, metadata)

            current_stream.wait_stream(self.transfer_stream)
            self.transfer_complete.synchronize()

            self._tensor_cache = tensors
            return self._tensors_to_features(tensors)

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Error distributing features: {str(e)}")

    def _tensors_to_features(self, tensors: Dict[str, torch.Tensor]) -> MarketFeatures:
        return MarketFeatures(
            ohlvc=tensors['ohlvc'],
            returns=tensors['returns'],
            sma=tensors['sma'],
            ema=tensors['ema'],
            rsi=tensors['rsi'],
            macd=tensors['macd'],
            reddit_sentiment=tensors['reddit_sentiment'],
            news_sentiment=tensors['news_sentiment'],
            combined_sentiment=tensors['combined_sentiment'],
            day=tensors['day'],
            month=tensors['month']
        )

    def cleanup(self) -> None:
        self._tensor_cache.clear()
        self._pinned_memory_pool.clear()
        self._feature_chunks.clear()
        torch.cuda.empty_cache()
        self.features = None
