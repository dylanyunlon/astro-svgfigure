/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

// [ASTRO-TRANSPORT] IntraDispatcher → in-process SVG cell direct-publish (M108-M110)
// IntraDispatcher routes cell messages within one process epoch boundary:
//   ChannelChain → ordered list of cell subscribers for a given SVG element channel
//   Zero-copy delivery → SVG cell data referenced by pointer, no serialization overhead

#include "cyber/transport/dispatcher/intra_dispatcher.h"

#include <cstdio>

namespace apollo {
namespace cyber {
namespace transport {

IntraDispatcher::IntraDispatcher() {
  chain_.reset(new ChannelChain());
  // [ASTRO-TRANSPORT] intra-process cell channel chain init (M108-M110)
  // ChannelChain is the ASTRO in-process pubsub backbone for SVG cell direct-dispatch.
  // Each channel in the chain corresponds to one SVG element type subscription group.
  fprintf(stderr, "[ASTRO-TRANSPORT] IntraDispatcher::ctor  channelChain=%p\n",
    (void*)chain_.get());
}

IntraDispatcher::~IntraDispatcher() {
  fprintf(stderr, "[ASTRO-TRANSPORT] IntraDispatcher::dtor  channelChain released\n");
}

}  // namespace transport
}  // namespace cyber
}  // namespace apollo
