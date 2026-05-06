import { registerComponents } from '@mutgui/core';

import './components/ChatInput.css';
import { ChatInput } from './components/ChatInput';

registerComponents({
  __name__: 'mutagent',
  ChatInput,
});

export { ChatInput };
