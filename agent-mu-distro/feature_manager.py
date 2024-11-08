import pickle
import torch

from typing import Optional, Dict

from config import MarketFeatures

class DistributedFeatureManager:
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

        self.compute_stream = torch.cuda.Stream()
        self.transfer_stream = torch.cuda.Stream()
        self.transfer_complete = torch.cuda.Event()
        self.device = torch.device(f'cuda:{rank}')

        self.features: Optional[MarketFeatures] = None
        self._tensor_cache: Dict[str, torch.Tensor] = {}

    def distribute_features(self, features: Optional[MarketFeatures] = None) -> MarketFeatures:
        try:
            current_stream = torch.cuda.current_stream()
            self.transfer_stream.wait_stream(current_stream)

            with torch.cuda.stream(self.transfer_stream):
                if self.rank == 0:
                    if features is None:
                        raise ValueError("Rank 0 must provide features")
                    self.features = features
                    feature_tensors = self._prepare_features(features)
                else:
                    feature_tensors = self._create_empty_tensors()

                for name, tensor in feature_tensors.items():
                    torch.distributed.broadcast(tensor, src=0)

                self.transfer_complete.record(self.transfer_stream)

            current_stream.wait_stream(self.transfer_stream)
            self.transfer_complete.synchronize()

            return self._tensors_to_features(feature_tensors)

        except Exception as e:
            self.cleanup()
            raise

    def _prepare_features(self, features: MarketFeatures) -> Dict[str, torch.Tensor]:
        try:
            return {
                'ohlvc': features.ohlvc.pin_memory().to(self.device, non_blocking=True),
                'returns': features.returns.pin_memory().to(self.device, non_blocking=True),
                'sma': features.sma.pin_memory().to(self.device, non_blocking=True),
                'ema': features.ema.pin_memory().to(self.device, non_blocking=True),
                'rsi': features.rsi.pin_memory().to(self.device, non_blocking=True),
                'macd': features.macd.pin_memory().to(self.device, non_blocking=True),
                'reddit_sentiment': features.reddit_sentiment.pin_memory().to(self.device, non_blocking=True),
                'news_sentiment': features.news_sentiment.pin_memory().to(self.device, non_blocking=True),
                'combined_sentiment': features.combined_sentiment.pin_memory().to(self.device, non_blocking=True),
                'day': features.day.pin_memory().to(self.device, non_blocking=True),
                'month': features.month.pin_memory().to(self.device, non_blocking=True)
            }
        except Exception as e:
            raise RuntimeError(f"Error preparing features: {str(e)}")

    def _create_empty_tensors(self) -> Dict[str, torch.Tensor]:
        try:
            sizes = None
            if self.rank == 0:
                if self.features is None:
                    raise ValueError("Features not set on rank 0")
                sizes = {
                    'ohlvc': list(self.features.ohlvc.shape),
                    'returns': list(self.features.returns.shape),
                    'sma': list(self.features.sma.shape),
                    'ema': list(self.features.ema.shape),
                    'rsi': list(self.features.rsi.shape),
                    'macd': list(self.features.macd.shape),
                    'reddit_sentiment': list(self.features.reddit_sentiment.shape),
                    'news_sentiment': list(self.features.news_sentiment.shape),
                    'combined_sentiment': list(self.features.combined_sentiment.shape),
                    'day': list(self.features.day.shape),
                    'month': list(self.features.month.shape)
                }

            sizes = self._broadcast_object(sizes)

            return {
                name: torch.empty(sizes[name], dtype=torch.float32, device=self.device)
                for name in sizes.keys()
            }
        except Exception as e:
            raise RuntimeError(f"Error creating empty tensors: {str(e)}")

    @staticmethod
    def _broadcast_object(obj: Optional[Dict[str, list[int]]] = None) -> Dict[str, list[int]]:
        if torch.distributed.get_rank() == 0:
            buffer = pickle.dumps(obj)
            size = torch.LongTensor([len(buffer)])
        else:
            size = torch.LongTensor([0])

        torch.distributed.broadcast(size, src=0)

        if torch.distributed.get_rank() == 0:
            buffer_tensor = torch.ByteTensor(list(buffer))
        else:
            buffer_tensor = torch.ByteTensor(size.item())

        torch.distributed.broadcast(buffer_tensor, src=0)

        if torch.distributed.get_rank() != 0:
            buffer = bytes(buffer_tensor.tolist())
            obj = pickle.loads(buffer)

        return obj

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

    def cleanup(self):
        self._tensor_cache.clear()
        torch.cuda.empty_cache()
        self.features = None
